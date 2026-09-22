"""Authentication and request throttling for one shared company workspace."""

import hashlib
import hmac
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Principal:
    name: str
    role: str


def parse_keys(raw):
    keys = json.loads(raw or "{}")
    if not isinstance(keys, dict):
        raise ValueError("RELAY_API_KEYS must be a JSON object.")
    for entry in keys.values():
        if (
            not isinstance(entry, dict)
            or len(entry.get("key", "")) < 24
            or entry.get("role") not in {"admin", "agent"}
        ):
            raise ValueError(
                "Each access key needs at least 24 characters and role admin or agent."
            )
    if len({entry["key"] for entry in keys.values()}) != len(keys):
        raise ValueError("Each user must have a unique API key.")
    return keys


def authenticate(header, keys):
    token = header[7:] if header.startswith("Bearer ") else ""
    for name, entry in keys.items():
        if hmac.compare_digest(token.encode(), entry["key"].encode()):
            return Principal(name, entry["role"])
    return None


class RateLimiter:
    """Per-process sliding window; a shared gateway is needed for multiple workers."""

    def __init__(self, limit=30, window=60):
        self.limit, self.window = limit, window
        self.hits = defaultdict(deque)

    def allow(self, identity):
        now = time.monotonic()
        key = hashlib.sha256(identity.encode()).hexdigest()
        if len(self.hits) >= 2000:
            self.hits = defaultdict(
                deque,
                {k: v for k, v in self.hits.items() if v and v[-1] > now - self.window},
            )
            if len(self.hits) >= 2000 and key not in self.hits:
                return False
        hits = self.hits[key]
        while hits and hits[0] <= now - self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            return False
        hits.append(now)
        return True
