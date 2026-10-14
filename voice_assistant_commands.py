"""Deterministic live-data handlers for Spatial Desktop voice requests."""
from __future__ import annotations

import os
import platform
import re
from datetime import datetime


def live_answer(text: str) -> tuple[str, str] | None:
    """Return (kind, response) for requests that must use real local data."""
    clean = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    clean = " ".join(clean.split())
    if re.search(r"\b(what time is it|whats the time|what s the time|what is the time|tell me the time|current time|time is it)\b", clean):
        now = datetime.now().astimezone()
        clock = now.strftime("%I:%M %p").lstrip("0")
        return "time", "It is " + clock + "."
    if re.search(r"\b(what is the date|whats the date|what s the date|what date is it|today s date|what day is it|what day is today|current date|tell me the date)\b", clean):
        now = datetime.now().astimezone()
        return "date", "Today is " + now.strftime("%A, %d %B %Y") + "."
    if re.search(r"\b(system info|system information|computer specs|pc specs|hardware information|how much ram|memory usage|cpu usage|system status)\b", clean):
        details = [f"You are running {platform.system()} {platform.release()}."]
        details.append(f"The machine architecture is {platform.machine()}.")
        try:
            import psutil
            mem = psutil.virtual_memory()
            details.append(f"Memory usage is {mem.percent} percent, with {mem.available / (1024**3):.1f} gigabytes available.")
            details.append(f"CPU usage is {psutil.cpu_percent(interval=0.15)} percent.")
        except Exception:
            details.append(f"There are {os.cpu_count() or 'an unknown number of'} logical CPU cores.")
        return "system", " ".join(details)
    return None


def file_query(text: str) -> str | None:
    """Extract a query only from clear spoken local-file search requests."""
    clean = " ".join((text or "").strip().split())
    low = clean.lower()
    if low in {"search files", "find files", "open files", "file search", "search bar"}:
        return None
    prefixes = (
        "search files for ", "find files for ", "search my files for ",
        "find my files for ", "search for ", "find ", "look for ",
        "locate ", "show me ", "where is ",
    )
    query = None
    for prefix in prefixes:
        if low.startswith(prefix) and len(clean) > len(prefix):
            query = clean[len(prefix):].strip()
            break
    if not query:
        return None
    query_words = query.split()
    if query_words and query_words[0].lower() in {"file", "folder", "document", "photo", "picture"}:
        query_words = query_words[1:]
    if query_words and query_words[0].lower() in {"named", "called"}:
        query_words = query_words[1:]
    result = " ".join(query_words).strip()
    return result or None
