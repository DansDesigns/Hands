"""Freehand local file index and user-selected search roots.

Standard-library only. Optional mutagen supplies richer music tags.
No network access, no file modifications, and no automatic scanning of other drives.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import threading
import time
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher

IGNORED_DIRECTORIES = {
    '.cache', '.git', '.hg', '.svn', '.venv', 'venv', 'node_modules',
    '__pycache__', '.tox', '.mypy_cache', '.pytest_cache', '$recycle.bin',
    'system volume information', 'appdata', '.local', '.npm', '.cargo',
    '.rustup', '.gradle', '.next', 'proc', 'sys', 'dev', 'run', 'tmp',
}
AUDIO_EXT = {'.mp3', '.flac', '.m4a', '.aac', '.ogg', '.opus', '.wav', '.wma'}
VIDEO_EXT = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.m4v', '.webm', '.ts', '.mpg', '.mpeg'}
EPISODE_PATTERN = re.compile(r'(?i)\bS(\d{1,2})[ ._\-]*E(\d{1,3})\b|\b(\d{1,2})x(\d{1,3})\b')


def _norm_text(value):
    txt = unicodedata.normalize('NFKD', str(value or ''))
    txt = ''.join(c for c in txt if not unicodedata.combining(c))
    return re.sub(r'\s+', ' ', re.sub(r'[^\w]+', ' ', txt.casefold())).strip()


def _folder(path):
    return str(Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve(strict=False))


def _key(path):
    return os.path.normcase(os.path.normpath(path))


def _contained(child, parent):
    try:
        return os.path.commonpath([_key(child), _key(parent)]) == _key(parent)
    except ValueError:
        return False


def _data_dir():
    if sys.platform.startswith('win'):
        return Path(os.environ.get('LOCALAPPDATA') or (Path.home() / 'AppData' / 'Local')) / 'Freehand'
    return Path(os.environ.get('XDG_DATA_HOME') or (Path.home() / '.local' / 'share')) / 'freehand'


class SearchIndex:
    """Threaded, incremental local-file index with persistent selected roots."""

    def __init__(self, data_directory=None, interval=600):
        self.data_dir = Path(data_directory) if data_directory else _data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.data_dir / 'search_folders.json'
        self.database = self.data_dir / 'search_index.sqlite3'
        self.interval = max(30, int(interval))
        self._mutex = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._state = {'scanning': False, 'count': 0, 'scanned': 0,
                       'root': '', 'error': '', 'last_completed': None}
        self._scopes, self._automatic = self._load_config()

    def _load_config(self):
        home = _folder(Path.home())
        try:
            data = json.loads(self.config_path.read_text(encoding='utf-8'))
            roots = data.get('roots', [])
            selected = []
            for entry in roots:
                if not isinstance(entry, dict) or not isinstance(entry.get('path'), str):
                    continue
                normalized = _folder(entry['path'])
                if not any(_key(x['path']) == _key(normalized) for x in selected):
                    selected.append({'path': normalized, 'enabled': bool(entry.get('enabled', True))})
            if not any(_key(r['path']) == _key(home) for r in selected):
                selected.insert(0, {'path': home, 'enabled': True})
            return selected, bool(data.get('automatic', True))
        except (OSError, ValueError, TypeError):
            return [{'path': home, 'enabled': True}], True

    def _save(self):
        document = {'version': 1, 'automatic': self._automatic, 'roots': self._scopes}
        tmp = self.config_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding='utf-8')
        os.replace(tmp, self.config_path)

    def get_scopes(self):
        with self._mutex:
            return [dict(root) for root in self._scopes]

    def automatic(self):
        with self._mutex:
            return self._automatic

    def set_automatic(self, enabled):
        with self._mutex:
            self._automatic = bool(enabled)
            self._save()
        if enabled:
            self.request_scan()

    def add_root(self, path):
        directory = _folder(path)
        if not os.path.isdir(directory):
            raise ValueError('Selected folder does not exist: ' + directory)
        with self._mutex:
            for root in self._scopes:
                if _key(root['path']) == _key(directory):
                    root['enabled'] = True
                    self._save()
                    self.request_scan()
                    return False
            self._scopes.append({'path': directory, 'enabled': True})
            self._save()
        self.request_scan()
        return True

    def set_enabled(self, path, enabled):
        with self._mutex:
            for root in self._scopes:
                if _key(root['path']) == _key(_folder(path)):
                    root['enabled'] = bool(enabled)
                    self._save()
                    self.request_scan()
                    return
        raise KeyError(path)

    def remove_root(self, path):
        directory = _folder(path)
        if _key(directory) == _key(_folder(Path.home())):
            raise ValueError('The default home folder can be disabled but not removed.')
        with self._mutex:
            before = len(self._scopes)
            self._scopes = [r for r in self._scopes if _key(r['path']) != _key(directory)]
            if len(self._scopes) == before:
                return False
            self._save()
        self.request_scan()
        return True

    def status(self):
        with self._mutex:
            return dict(self._state)

    def request_scan(self):
        self._wake.set()

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name='FreehandFileIndexer', daemon=True)
            self._thread.start()
            self.request_scan()

    def close(self):
        self._stop.set()
        self._wake.set()
        if self._thread is not None and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)

    @staticmethod
    def _connect(path):
        conn = sqlite3.connect(str(path), timeout=8)
        conn.execute('PRAGMA busy_timeout=8000')
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('''CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, root TEXT NOT NULL,
            name TEXT NOT NULL, parent TEXT NOT NULL, meta TEXT NOT NULL DEFAULT '',
            kind TEXT NOT NULL, size INTEGER, mtime_ns INTEGER, seen INTEGER NOT NULL DEFAULT 0
        )''')
        conn.execute('CREATE INDEX IF NOT EXISTS files_root_idx ON files(root)')
        try:
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(name, parent, meta, content='files', content_rowid='id')")
            conn.executescript('''CREATE TRIGGER IF NOT EXISTS files_fts_ai AFTER INSERT ON files BEGIN
                INSERT INTO files_fts(rowid,name,parent,meta) VALUES(new.id,new.name,new.parent,new.meta); END;
            CREATE TRIGGER IF NOT EXISTS files_fts_ad AFTER DELETE ON files BEGIN
                INSERT INTO files_fts(files_fts,rowid,name,parent,meta) VALUES('delete',old.id,old.name,old.parent,old.meta); END;
            CREATE TRIGGER IF NOT EXISTS files_fts_au AFTER UPDATE OF name,parent,meta ON files
                WHEN new.name != old.name OR new.parent != old.parent OR new.meta != old.meta BEGIN
                INSERT INTO files_fts(files_fts,rowid,name,parent,meta) VALUES('delete',old.id,old.name,old.parent,old.meta);
                INSERT INTO files_fts(rowid,name,parent,meta) VALUES(new.id,new.name,new.parent,new.meta); END;''')
        except sqlite3.OperationalError:
            pass  # SQLite without FTS5 can use the LIKE fallback.
        return conn

    def _run(self):
        while not self._stop.is_set():
            woke = self._wake.wait(self.interval)
            self._wake.clear()
            if self._stop.is_set():
                return
            if not woke and not self.automatic():
                continue
            try:
                self._scan()
            except Exception as exc:
                with self._mutex:
                    self._state['error'] = f'{type(exc).__name__}: {exc}'
                    self._state['scanning'] = False

    @staticmethod
    def _media_meta(name, fullpath, kind):
        normalized = _norm_text(name)
        episode = EPISODE_PATTERN.search(name)
        chunks = [normalized]
        if episode:
            s = int(episode.group(1) or episode.group(3))
            e = int(episode.group(2) or episode.group(4))
            chunks += [f'season {s} episode {e}', f's{s:02d}e{e:02d}']
        if kind == 'audio':
            try:
                from mutagen import File as AudioFile  # optional; imported lazily
                tags = AudioFile(fullpath, easy=True)
                if tags:
                    for key in ('title', 'artist', 'album', 'genre', 'albumartist'):
                        val = tags.get(key)
                        if val:
                            chunks.extend(_norm_text(v) for v in val[:4])
            except Exception:
                pass
        return ' '.join(chunks)[:2000]

    def _scan(self):
        with self._mutex:
            scopes = [x['path'] for x in self._scopes if x['enabled']]
            self._state.update(scanning=True, scanned=0, error='', root='')
        # Nested selected folders are covered by the containing root.
        scopes = sorted(set(scopes), key=lambda p: (len(p), p.casefold()))
        effective = []
        for path in scopes:
            if not any(_contained(path, other) for other in effective):
                effective.append(path)
        conn = self._connect(self.database)
        generation = time.time_ns()
        processed = 0
        try:
            with conn:
                if effective:
                    conn.execute('DELETE FROM files WHERE root NOT IN (' + ','.join('?' * len(effective)) + ')', effective)
                else:
                    conn.execute('DELETE FROM files')
            for root in effective:
                if self._stop.is_set() or self._wake.is_set():
                    break
                with self._mutex:
                    self._state['root'] = root
                if not os.path.isdir(root):
                    continue  # Retain previous index for offline/removable folders.
                with conn:
                    conn.execute('UPDATE files SET seen=-1 WHERE root=?', (root,))
                pending = [root]
                aborted = False
                while pending:
                    if self._stop.is_set() or self._wake.is_set():
                        aborted = True
                        break
                    directory = pending.pop()
                    try:
                        iterator = os.scandir(directory)
                    except OSError:
                        continue
                    with iterator:
                        for entry in iterator:
                            if self._stop.is_set() or self._wake.is_set():
                                aborted = True
                                break
                            try:
                                if entry.is_symlink():
                                    continue
                                is_dir = entry.is_dir(follow_symlinks=False)
                                if is_dir and entry.name.casefold() in IGNORED_DIRECTORIES:
                                    continue
                                stat = entry.stat(follow_symlinks=False)
                                filepath = entry.path
                                kind = ('folder' if is_dir else 'audio' if Path(entry.name).suffix.lower() in AUDIO_EXT
                                        else 'video' if Path(entry.name).suffix.lower() in VIDEO_EXT else 'file')
                                old = conn.execute('SELECT size,mtime_ns,meta FROM files WHERE path=?', (filepath,)).fetchone()
                                meta = old[2] if old and old[:2] == (stat.st_size, stat.st_mtime_ns) else (
                                    '' if is_dir else self._media_meta(entry.name, filepath, kind))
                                conn.execute('''INSERT INTO files(path,root,name,parent,meta,kind,size,mtime_ns,seen)
                                    VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET
                                    root=excluded.root,name=excluded.name,parent=excluded.parent,meta=excluded.meta,
                                    kind=excluded.kind,size=excluded.size,mtime_ns=excluded.mtime_ns,seen=excluded.seen''',
                                    (filepath, root, entry.name, directory, meta, kind, stat.st_size, stat.st_mtime_ns, generation))
                                processed += 1
                                if is_dir:
                                    pending.append(filepath)
                                if processed % 300 == 0:
                                    conn.commit()
                                    with self._mutex:
                                        self._state['scanned'] = processed
                            except (OSError, ValueError):
                                continue
                    if aborted:
                        break
                conn.commit()
                if not aborted:
                    with conn:
                        conn.execute('DELETE FROM files WHERE root=? AND seen=-1', (root,))
            total = conn.execute('SELECT COUNT(*) FROM files').fetchone()[0]
            with self._mutex:
                self._state.update(count=total, scanned=processed,
                                   last_completed=time.time() if not self._stop.is_set() else self._state['last_completed'])
        finally:
            with self._mutex:
                self._state['scanning'] = False
                self._state['root'] = ''
            conn.close()

    def search(self, query, limit=30, kind=None):
        """Return matching local files: [{'path', 'name', 'kind', 'score'}, ...].

        Retrieval is deterministic; D1 should rerank only plausible candidates,
        with exact season/episode constraints enforced by the caller.
        """
        words = _norm_text(query).split()
        if not words or not self.database.exists():
            return []
        conn = self._connect(self.database)
        conn.row_factory = sqlite3.Row
        maximum = max(1, min(100, int(limit)))
        results = []
        try:
            clauses = ' AND kind=?' if kind else ''
            params = [kind] if kind else []
            try:
                match = ' OR '.join('"' + w.replace('"', '') + '"*' for w in words[:10])
                rows = conn.execute('''SELECT f.path, f.name, f.kind, f.parent, f.meta,
                                          bm25(files_fts) AS rank
                                   FROM files_fts JOIN files f ON f.id=files_fts.rowid
                                   WHERE files_fts MATCH ?''' + clauses +
                                    ' ORDER BY rank LIMIT ?', [match]+params+[maximum*10]).fetchall()
            except sqlite3.OperationalError:
                like = '%' + words[0] + '%'
                rows = conn.execute('SELECT path,name,kind,parent,meta,0 AS rank FROM files WHERE lower(name) LIKE ? OR lower(parent) LIKE ? LIMIT ?',
                                    (like, like, maximum*10)).fetchall()
            for r in rows:
                title = _norm_text(r['name'] + ' ' + r['parent'] + ' ' + r['meta'])
                coverage = sum(w in title for w in words)/len(words)
                similarity = SequenceMatcher(None, _norm_text(query), title[:250]).ratio()
                score = round(0.83*coverage + 0.17*similarity, 4)
                results.append({'path': r['path'], 'name': r['name'], 'kind': r['kind'], 'score': score})
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:maximum]
        finally:
            conn.close()
