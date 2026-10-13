#!/usr/bin/env python3
"""Fast, local file search for Spatial Desktop.

No embedding model or LLM is used. A background indexer stores file names,
paths and small text/document excerpts in SQLite FTS5. Ranking combines BM25,
filename/path weighting, phrase matches and a small local synonym map.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

APP_DATA = Path.home() / ".spatial_desktop"
DB_PATH = APP_DATA / "local_file_search.sqlite3"
ROOTS_PATH = APP_DATA / "search_roots.json"
MAX_TEXT_BYTES = 2 * 1024 * 1024
MAX_INDEXED_CHARS = 18_000

TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".py", ".pyi", ".json", ".csv", ".tsv", ".html", ".htm",
    ".xml", ".yaml", ".yml", ".ini", ".cfg", ".toml", ".bat", ".cmd", ".ps1", ".sh",
    ".js", ".ts", ".jsx", ".tsx", ".css", ".scss", ".c", ".h", ".cc", ".cpp", ".hpp",
    ".java", ".kt", ".go", ".rs", ".sql", ".log", ".conf", ".properties", ".tex", ".rtf",
    ".srt", ".vtt", ".ini", ".org", ".adoc", ".markdown",
}
DOCUMENT_EXTENSIONS = {".pdf", ".docx"}
SUPPORTED_CONTENT = TEXT_EXTENSIONS | DOCUMENT_EXTENSIONS
SKIP_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules", "__pycache__",
    ".cache", ".npm", ".cargo", ".rustup", ".local", "appdata", "$recycle.bin",
    "system volume information", "windows", "program files", "program files (x86)",
    "dist", "build", ".idea", ".vscode", "models", "target",
}
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "did", "do", "for",
    "from", "how", "i", "in", "into", "is", "it", "me", "my", "of", "on", "or",
    "please", "the", "that", "there", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "with", "you", "your", "find", "look", "search", "file",
    "files", "document", "documents", "about", "some", "something",
}
# Small, deterministic vocabulary expansion: helpful for ordinary file search,
# but intentionally far smaller and cheaper than a learned embedding model.
SYNONYMS = {
    "photo": ("picture", "image", "photograph"),
    "photos": ("pictures", "images", "photographs"),
    "picture": ("photo", "image"),
    "image": ("photo", "picture"),
    "screenshot": ("screen", "capture"),
    "meeting": ("call", "discussion", "minutes"),
    "minutes": ("meeting", "notes"),
    "recording": ("audio", "voice", "transcript"),
    "recordings": ("audio", "voice", "transcripts"),
    "voice": ("audio", "recording", "speech"),
    "invoice": ("bill", "receipt", "payment"),
    "receipt": ("invoice", "bill", "purchase"),
    "crash": ("error", "exception", "log"),
    "error": ("crash", "exception", "log"),
    "settings": ("config", "configuration", "preferences"),
    "configuration": ("config", "settings", "preferences"),
    "setup": ("install", "installer", "installation"),
    "installer": ("setup", "install", "installation"),
    "notes": ("memo", "summary", "minutes"),
    "memo": ("notes", "summary"),
    "presentation": ("slides", "powerpoint"),
    "slides": ("presentation", "powerpoint"),
    "video": ("clip", "movie"),
    "movie": ("video", "clip"),
    "resume": ("cv", "curriculum"),
    "cv": ("resume", "curriculum"),
    "calibration": ("setup", "alignment", "configuration"),
    "sensitivity": ("responsiveness", "speed"),
}


def _norm_path(path: Path | str) -> str:
    return str(Path(path).expanduser().resolve())


def default_roots() -> list[str]:
    """Choose common personal folders; never index the whole home folder by default."""
    home = Path.home()
    candidates = [
        home / "Desktop", home / "Documents", home / "Downloads", home / "Pictures",
        home / "Videos", home / "Music", home / "OneDrive",
    ]
    existing: list[Path] = []
    for candidate in candidates:
        try:
            if not candidate.is_dir():
                continue
            resolved = candidate.resolve()
        except OSError:
            continue
        if any(resolved == prev or prev in resolved.parents for prev in existing):
            continue
        # If a broad cloud-sync folder appears later, prefer it over its already
        # collected child folders to avoid indexing the same files twice.
        existing = [prev for prev in existing if resolved not in prev.parents]
        existing.append(resolved)
    return [str(p) for p in existing]


def tokens_for_query(query: str) -> list[str]:
    raw = [token.lower() for token in re.findall(r"[\w]{2,}", query, flags=re.UNICODE)]
    useful = [token for token in raw if token not in STOP_WORDS]
    if not useful:
        useful = raw
    terms: list[str] = []
    for token in useful:
        if token not in terms:
            terms.append(token)
        for synonym in SYNONYMS.get(token, ()):
            if synonym not in terms:
                terms.append(synonym)
    return terms[:18]


def extract_text(path: Path) -> str:
    """Read bounded text from common documents. Unsupported file types use name/path only."""
    suffix = path.suffix.lower()
    try:
        if path.stat().st_size > MAX_TEXT_BYTES:
            return ""
    except OSError:
        return ""
    try:
        if suffix in TEXT_EXTENSIONS:
            raw = path.read_bytes()
            if b"\x00" in raw[:4096]:
                return ""
            text = raw.decode("utf-8-sig", errors="replace")
            if suffix in {".html", ".htm", ".xml"}:
                text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
                text = re.sub(r"(?s)<[^>]+>", " ", text)
            return " ".join(text.split())[:MAX_INDEXED_CHARS]
        if suffix == ".pdf":
            from pypdf import PdfReader
            parts = []
            for page in PdfReader(str(path)).pages[:30]:
                parts.append(page.extract_text() or "")
                if sum(map(len, parts)) >= MAX_INDEXED_CHARS:
                    break
            return " ".join(" ".join(parts).split())[:MAX_INDEXED_CHARS]
        if suffix == ".docx":
            from docx import Document
            doc = Document(str(path))
            return " ".join(p.text for p in doc.paragraphs if p.text)[:MAX_INDEXED_CHARS]
    except Exception:
        return ""
    return ""


def iter_files(root: Path) -> Iterable[Path]:
    def onerror(_exc):
        return None

    for current, dirs, files in os.walk(root, topdown=True, followlinks=False, onerror=onerror):
        dirs[:] = [d for d in dirs if d.lower() not in SKIP_DIRS and not d.startswith(".")]
        for name in files:
            if name.startswith("."):
                continue
            path = Path(current) / name
            try:
                if path.is_file():
                    yield path
            except OSError:
                continue


@dataclass(frozen=True)
class SearchResult:
    path: str
    name: str
    parent: str
    snippet: str
    score: float
    suffix: str


class LocalFileIndex:
    """Incremental SQLite FTS5 index; all content and searches remain local."""

    def __init__(self, db_path: Path | str | None = None, roots_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.roots_path = Path(roots_path) if roots_path else ROOTS_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.roots_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.roots_path.exists():
            self._save_roots(default_roots())
        self.fts_available = True
        conn = self.connect()
        try:
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                pass
            conn.execute("""CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                root TEXT NOT NULL,
                name TEXT NOT NULL,
                parent TEXT NOT NULL,
                suffix TEXT NOT NULL,
                size INTEGER NOT NULL,
                mtime REAL NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                seen_at REAL NOT NULL DEFAULT 0
            )""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_files_root ON files(root)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_files_name ON files(name COLLATE NOCASE)")
            try:
                conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS file_fts USING fts5(
                    name, parent, content, tokenize='porter unicode61 remove_diacritics 2')""")
            except sqlite3.OperationalError:
                self.fts_available = False
            conn.commit()
        finally:
            conn.close()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=4.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=4000")
        return conn

    def roots(self) -> list[str]:
        try:
            data = json.loads(self.roots_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            data = default_roots()
        if not isinstance(data, list):
            data = default_roots()
        result = []
        for item in data:
            if not isinstance(item, str):
                continue
            path = Path(item).expanduser()
            if path.is_dir():
                value = _norm_path(path)
                if value not in result:
                    result.append(value)
        return result

    def _save_roots(self, roots: list[str]) -> None:
        try:
            self.roots_path.write_text(json.dumps(roots, indent=2) + "\n", encoding="utf-8")
        except OSError:
            pass

    def add_root(self, folder: str | Path) -> tuple[bool, str]:
        path = Path(folder).expanduser()
        if not path.is_dir():
            return False, "Choose an existing folder."
        value = _norm_path(path)
        roots = [Path(r) for r in self.roots()]
        for existing in roots:
            if value == str(existing) or existing in Path(value).parents:
                return False, "That folder is already covered by an indexed location."
        # If a broader new root contains configured roots, replace those roots.
        roots = [r for r in roots if r not in Path(value).parents]
        roots.append(Path(value))
        roots = sorted(set(str(r) for r in roots))
        self._save_roots(roots)
        return True, f"Added {value}"

    def index_roots(self, roots: list[str] | None = None,
                    progress: Callable[[str], None] | None = None,
                    cancelled: Callable[[], bool] | None = None) -> tuple[int, int]:
        roots = roots if roots is not None else self.roots()
        report = progress or (lambda _message: None)
        is_cancelled = cancelled or (lambda: False)
        seen_total = changed_total = 0
        conn = self.connect()
        fts_ok = self.fts_available
        if fts_ok:
            try:
                conn.execute("SELECT count(*) FROM file_fts").fetchone()
            except sqlite3.OperationalError:
                fts_ok = False
        try:
            for root_text in roots:
                root = Path(root_text)
                if not root.is_dir():
                    continue
                root_text = _norm_path(root)
                scan_started = time.time()
                seen = changed = 0
                had_errors = False
                for path in iter_files(root):
                    if is_cancelled():
                        had_errors = True  # do not prune stale records for an incomplete scan
                        break
                    try:
                        stat = path.stat()
                        abs_path = _norm_path(path)
                    except OSError:
                        had_errors = True
                        continue
                    name = path.name
                    parent = str(path.parent)
                    suffix = path.suffix.lower()
                    old = conn.execute("SELECT id,size,mtime FROM files WHERE path=?", (abs_path,)).fetchone()
                    if old and old["size"] == stat.st_size and abs(old["mtime"] - stat.st_mtime) < 0.0001:
                        conn.execute("UPDATE files SET seen_at=?,root=? WHERE id=?", (scan_started, root_text, old["id"]))
                        seen += 1
                    else:
                        text = extract_text(path)
                        if old:
                            rowid = old["id"]
                            conn.execute("""UPDATE files SET root=?,name=?,parent=?,suffix=?,size=?,mtime=?,content=?,seen_at=?
                                WHERE id=?""", (root_text, name, parent, suffix, stat.st_size, stat.st_mtime,
                                                  text, scan_started, rowid))
                            if fts_ok:
                                conn.execute("DELETE FROM file_fts WHERE rowid=?", (rowid,))
                        else:
                            cur = conn.execute("""INSERT INTO files(path,root,name,parent,suffix,size,mtime,content,seen_at)
                                VALUES(?,?,?,?,?,?,?,?,?)""", (abs_path, root_text, name, parent, suffix,
                                    stat.st_size, stat.st_mtime, text, scan_started))
                            rowid = cur.lastrowid
                        if fts_ok:
                            conn.execute("INSERT INTO file_fts(rowid,name,parent,content) VALUES(?,?,?,?)",
                                         (rowid, name, parent, text))
                        seen += 1
                        changed += 1
                    if seen % 60 == 0:
                        conn.commit()
                        report(f"Indexing {root.name}: {seen:,} files scanned, {changed:,} updated")
                if is_cancelled():
                    break
                if not had_errors:
                    stale = conn.execute("SELECT id FROM files WHERE root=? AND seen_at<?",
                                          (root_text, scan_started)).fetchall()
                    for row in stale:
                        if fts_ok:
                            conn.execute("DELETE FROM file_fts WHERE rowid=?", (row["id"],))
                        conn.execute("DELETE FROM files WHERE id=?", (row["id"],))
                conn.commit()
                seen_total += seen
                changed_total += changed
                report(f"Indexed {seen:,} files in {root}")
        finally:
            conn.close()
        return seen_total, changed_total

    @staticmethod
    def _snippet(content: str, query_tokens: list[str], max_chars: int = 150) -> str:
        text = " ".join(content.split())
        if not text:
            return ""
        low = text.lower()
        positions = [low.find(token) for token in query_tokens if low.find(token) >= 0]
        start = max(0, min(positions) - 48) if positions else 0
        sample = text[start:start + max_chars]
        return ("…" if start else "") + sample + ("…" if start + max_chars < len(text) else "")

    def search(self, query: str, limit: int = 14) -> list[SearchResult]:
        query = " ".join(query.split()).strip()
        if len(query) < 2:
            return []
        terms = tokens_for_query(query)
        if not terms:
            return []
        original = [t for t in re.findall(r"[\w]{2,}", query.lower()) if t not in STOP_WORDS]
        original = original or terms[:]
        # FTS5 quoted-prefix OR terms are safe for arbitrary user text and support live typing.
        match = " OR ".join('"' + term.replace('"', '""') + '"*' for term in terms)
        conn = self.connect()
        candidates = []
        try:
            if self.fts_available:
                try:
                    rows = conn.execute("""SELECT f.id,f.path,f.name,f.parent,f.suffix,f.content,
                        bm25(file_fts, 9.0, 4.0, 1.0) AS rank
                        FROM file_fts JOIN files f ON f.id=file_fts.rowid
                        WHERE file_fts MATCH ? ORDER BY rank LIMIT 100""", (match,)).fetchall()
                except sqlite3.OperationalError:
                    rows = []
                for row in rows:
                    name_low = row["name"].lower()
                    parent_low = row["parent"].lower()
                    score = -float(row["rank"])
                    if query.lower() in name_low:
                        score += 8.0
                    elif query.lower() in parent_low:
                        score += 3.0
                    score += 2.4 * sum(1 for token in original if token in name_low) / max(1, len(original))
                    score += 0.65 * sum(1 for token in original if token in parent_low) / max(1, len(original))
                    if len(original) == 1:
                        ratio = difflib.SequenceMatcher(None, original[0], Path(row["name"]).stem.lower()).ratio()
                        score += max(0.0, ratio - 0.55) * 1.8
                    candidates.append((score, row))
            if not candidates:
                # Filename/path fallback also works if SQLite was built without FTS5.
                rows = conn.execute("SELECT id,path,name,parent,suffix,content FROM files ORDER BY mtime DESC LIMIT 12000").fetchall()
                qlow = query.lower()
                for row in rows:
                    name_low, parent_low = row["name"].lower(), row["parent"].lower()
                    token_hits = sum(1 for token in original if token in (name_low + " " + parent_low + " " + row["content"].lower()))
                    fuzzy = difflib.SequenceMatcher(None, qlow, name_low).ratio()
                    score = token_hits / max(1, len(original)) + fuzzy * 0.25
                    if qlow in name_low:
                        score += 4.0
                    elif qlow in parent_low:
                        score += 1.5
                    if score > 0.25:
                        candidates.append((score, row))
            candidates.sort(key=lambda item: item[0], reverse=True)
            results = []
            seen_paths = set()
            for score, row in candidates:
                path = row["path"]
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                snippet = self._snippet(row["content"], original)
                results.append(SearchResult(path, row["name"], row["parent"], snippet, score, row["suffix"]))
                if len(results) >= max(1, limit):
                    break
            return results
        finally:
            conn.close()

    def count(self) -> int:
        with self.connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM files").fetchone()
            return int(row[0]) if row else 0
