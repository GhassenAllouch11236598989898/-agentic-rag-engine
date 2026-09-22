"""Small, durable single-workspace store. Every conversation has an owner."""

import hashlib
import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


STOPWORDS = set(
    "a an the is are was were be been to of for and or in on with my our your their this that it can could would should how what when why do does did i we you please customer wants want need help about me us have has had if from as at northstar cloud business team confirm".split()
)


def words(text):
    return [
        w
        for w in re.findall(r"[a-z0-9]+", text.lower())
        if w not in STOPWORDS and len(w) > 1
    ]


class Workspace:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def initialize(self):
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, checksum TEXT UNIQUE NOT NULL,
                    content TEXT NOT NULL, created_at TEXT NOT NULL, kind TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
                    content TEXT NOT NULL, page INTEGER, position INTEGER NOT NULL);
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_search USING fts5(
                    chunk_id UNINDEXED, content, tokenize='porter unicode61');
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT REFERENCES sessions(id), role TEXT NOT NULL,
                    content TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS drafts (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, question TEXT NOT NULL,
                    payload TEXT NOT NULL, feedback TEXT, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
            """)

    def is_seeded(self):
        with self.connect() as db:
            return bool(
                db.execute(
                    "SELECT 1 FROM settings WHERE key='sample_seeded'"
                ).fetchone()
            )

    def mark_seeded(self):
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO settings VALUES ('sample_seeded','yes')")

    def ingest(self, title, pages, kind="uploaded"):
        content = "\n\n".join(text for _, text in pages)
        if len(content.strip()) < 20:
            raise ValueError(
                "No usable text found. Upload a text PDF, Markdown, or text file; scanned PDFs need OCR first."
            )
        checksum = hashlib.sha256(content.encode()).hexdigest()
        with self.connect() as db:
            existing = db.execute(
                "SELECT id FROM documents WHERE checksum=?", (checksum,)
            ).fetchone()
            if existing:
                return {
                    "document_id": existing["id"],
                    "duplicate": True,
                    "title": title,
                }
            doc_id = str(uuid.uuid4())
            db.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?)",
                (doc_id, title, checksum, content, now(), kind),
            )
            position = 0
            for page, text in pages:
                # Overlapping windows preserve short policy statements and page provenance.
                start = 0
                while start < len(text):
                    end = min(start + 1100, len(text))
                    if end < len(text):
                        boundary = text.rfind("\n", start + 500, end)
                        if boundary > start:
                            end = boundary
                    passage = text[start:end].strip()
                    if passage:
                        chunk_id = str(uuid.uuid4())
                        db.execute(
                            "INSERT INTO chunks VALUES (?,?,?,?,?)",
                            (chunk_id, doc_id, passage, page, position),
                        )
                        db.execute(
                            "INSERT INTO chunk_search VALUES (?,?)",
                            (chunk_id, title + "\n" + passage),
                        )
                        position += 1
                    if end == len(text):
                        break
                    start = max(start + 1, end - 140)
        return {
            "document_id": doc_id,
            "duplicate": False,
            "title": title,
            "chunks_created": position,
        }

    def documents(self):
        with self.connect() as db:
            return [
                dict(r)
                for r in db.execute("""SELECT d.id,d.title,d.kind,d.created_at,
                COUNT(c.id) chunk_count,LENGTH(d.content) characters FROM documents d
                LEFT JOIN chunks c ON c.document_id=d.id GROUP BY d.id ORDER BY d.created_at DESC""")
            ]

    def document(self, document_id):
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM documents WHERE id=?", (document_id,)
            ).fetchone()
            return dict(row) if row else None

    def delete_document(self, document_id):
        with self.connect() as db:
            db.execute(
                "DELETE FROM chunk_search WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id=?)",
                (document_id,),
            )
            return (
                db.execute("DELETE FROM documents WHERE id=?", (document_id,)).rowcount
                > 0
            )

    def search(self, query, limit=4):
        tokens = list(dict.fromkeys(words(query)))[:32]
        if not tokens:
            return []
        expression = " OR ".join('"' + t + '"' for t in tokens)
        with self.connect() as db:
            rows = db.execute(
                """SELECT c.*,d.title,bm25(chunk_search) rank
                FROM chunk_search JOIN chunks c ON c.id=chunk_search.chunk_id
                JOIN documents d ON d.id=c.document_id WHERE chunk_search MATCH ?
                ORDER BY rank LIMIT ?""",
                (expression, limit * 3),
            ).fetchall()
        results = []
        for row in rows:
            result = dict(row)
            # A single incidental keyword must not produce a confident draft.
            overlap = set(tokens) & set(
                words(result["content"] + " " + result["title"])
            )
            if len(overlap) < min(2, len(tokens)):
                continue
            result["score"] = round(-result.pop("rank"), 6)
            result["chunk_id"] = result.pop("id")
            results.append(result)
        return results[:limit]

    def session(self, owner, session_id=None):
        with self.connect() as db:
            if session_id:
                row = db.execute(
                    "SELECT owner FROM sessions WHERE id=?", (session_id,)
                ).fetchone()
                if not row or row["owner"] != owner:
                    raise LookupError("Conversation not found.")
                return session_id
            session_id = str(uuid.uuid4())
            db.execute(
                "INSERT INTO sessions VALUES (?,?,?)", (session_id, owner, now())
            )
            return session_id

    def history(self, owner, session_id, limit=10):
        self.session(owner, session_id)
        with self.connect() as db:
            rows = db.execute(
                """SELECT role,content FROM (
                SELECT id,role,content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?
                ) ORDER BY id ASC""",
                (session_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def save_draft(self, owner, session_id, question, payload):
        self.session(owner, session_id)
        with self.connect() as db:
            db.executemany(
                "INSERT INTO messages(session_id,role,content,created_at) VALUES (?,?,?,?)",
                [
                    (session_id, "user", question, now()),
                    (session_id, "assistant", payload["message"], now()),
                ],
            )
            db.execute(
                "INSERT INTO drafts VALUES (?,?,?,?,?,?)",
                (
                    payload["draft_id"],
                    owner,
                    question,
                    json.dumps(payload),
                    None,
                    now(),
                ),
            )

    def feedback(self, owner, draft_id, value):
        with self.connect() as db:
            return (
                db.execute(
                    "UPDATE drafts SET feedback=? WHERE id=? AND owner=?",
                    (value, draft_id, owner),
                ).rowcount
                > 0
            )

    def activity(self, owner):
        with self.connect() as db:
            rows = db.execute(
                "SELECT question,payload,feedback,created_at FROM drafts WHERE owner=? ORDER BY created_at DESC LIMIT 100",
                (owner,),
            ).fetchall()
            return [
                {
                    "question": r["question"],
                    "feedback": r["feedback"],
                    "created_at": r["created_at"],
                    **json.loads(r["payload"]),
                }
                for r in rows
            ]
