#!/usr/bin/env python3
"""
Context Memory for Hermes - persistent local context store.
Uses local embeddings + JSON storage with semantic search.
"""
import json
import os
import math
import hashlib
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ContextBlock:
    """A single unit of stored context."""
    id: str
    content: str
    source: str = ""
    tags: list = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    importance: float = 1.0  # 0-1, decayed over time
    embedding: list = field(default_factory=list)
    checksum: str = ""

    def __post_init__(self):
        if not self.checksum:
            self.checksum = hashlib.sha256(self.content.encode()).hexdigest()[:16]


class ContextStore:
    """Primary storage backend for context blocks."""
    
    def __init__(self, data_dir: str = "/home/ayan/context_memory/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.data_dir / "index.json"
        self.blocks: dict[str, ContextBlock] = {}
        self._load()
    
    def _load(self):
        """Load all saved blocks from disk."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                raw = json.load(f)
            for b in raw:
                cb = ContextBlock(**b)
                # Recompute checksum on load
                actual_cs = hashlib.sha256(cb.content.encode()).hexdigest()[:16]
                if actual_cs != cb.checksum:
                    cb.content = "[CORRUPTED OR MODIFIED]"
                    cb.checksum = actual_cs
                self.blocks[cb.id] = cb
    
    def save(self):
        """Persist index to disk."""
        # Decay importance before saving
        self._decay_importance()
        out = [asdict(b) for b in self.blocks.values()]
        tmp = str(self.index_file) + ".tmp"
        with open(tmp, 'w') as f:
            json.dump(out, f, indent=2)
        os.replace(tmp, self.index_file)
    
    def _decay_importance(self):
        """Halve importance per week of inactivity."""
        now = time.time()
        for b in self.blocks.values():
            age_days = (now - b.timestamp) / 86400
            weeks = age_days / 7
            b.importance = max(0.01, b.importance * (0.5 ** weeks))
    
    def add(self, content: str, source: str = "", tags: list = None,
            importance: float = 1.0) -> str:
        """Add a new context block. Returns ID."""
        if tags is None:
            tags = []
        block_id = hashlib.sha256(content.encode()).hexdigest()[:12]
        # Deduplicate by content checksum
        for existing in self.blocks.values():
            if existing.checksum == hashlib.sha256(content.encode()).hexdigest()[:16]:
                existing.access_count += 1
                existing.timestamp = time.time()
                self.save()
                return existing.id
        
        block = ContextBlock(
            id=block_id,
            content=content,
            source=source,
            tags=tags,
            importance=importance,
            checksum=hashlib.sha256(content.encode()).hexdigest()[:16],
        )
        self.blocks[block_id] = block
        self.save()
        return block_id
    
    def get(self, block_id: str) -> Optional[ContextBlock]:
        b = self.blocks.get(block_id)
        if b:
            b.access_count += 1
            b.timestamp = time.time()
        return b
    
    def delete(self, block_id: str) -> bool:
        if block_id in self.blocks:
            del self.blocks[block_id]
            self.save()
            return True
        return False
    
    def stats(self) -> dict:
        total = len(self.blocks)
        sources = {}
        for b in self.blocks.values():
            sources[b.source] = sources.get(b.source, 0) + 1
        return {
            "total_blocks": total,
            "sources": sources,
            "storage_path": str(self.index_file),
        }


class EmbeddingEngine:
    """
    Lightweight local embedding engine.
    Stable across process restarts (deterministic hash).
    Uses a combination of TF-IDF (if scikit-learn available) and
    character n-gram hashing into fixed-dimension space.
    """
    
    def __init__(self, dims: int = 256):
        self.dims = dims
        self._sklearn_vec = None
        self._sklearn_fitted = False
        self._try_sklearn()
    
    @staticmethod
    def _stable_hash(s: str) -> int:
        """Deterministic hash that is consistent across Python process restarts."""
        import hashlib
        return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16)
    
    def _try_sklearn(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._sklearn_vec = TfidfVectorizer(
                max_features=4096,
                ngram_range=(1, 2),
                stop_words="english",
            )
            self._sklearn_available = True
        except ImportError:
            self._sklearn_available = False
    
    def _char_ngrams(self, text: str, n: int = 3) -> list[str]:
        t = text.lower()
        return [t[i:i+n] for i in range(max(0, len(t) - n + 1))]
    
    def _hash_ngrams(self, tokens: list[str]) -> list[float]:
        vec = [0.0] * self.dims
        for tok in tokens:
            vec[self._stable_hash(tok) % self.dims] += 1.0
        return vec
    
    def _combine(self, vecs: list[list[float]]) -> list[float]:
        # Determine target size: concatenate and project to dims
        total_dim = sum(len(v) for v in vecs)
        merged = []
        for v in vecs:
            merged.extend(v)
        # If already exactly dims, just normalize
        if total_dim == self.dims:
            mag = math.sqrt(sum(x*x for x in merged)) or 1.0
            return [x / mag for x in merged]
        # Otherwise fold into dims buckets
        out = [0.0] * self.dims
        for i, v in enumerate(merged):
            out[i % self.dims] += v
        mag = math.sqrt(sum(x*x for x in out)) or 1.0
        return [x / mag for x in out]
    
    def embed(self, text: str) -> list[float]:
        words = [w.lower() for w in text.split()]
        word_vec = self._hash_ngrams(words)
        chars = self._char_ngrams(text, 3)
        char_vec = self._hash_ngrams(chars)
        
        if self._sklearn_fitted:
            sk_vec = self._sklearn_vec.transform([text]).toarray()[0].tolist()
            # Fold to dims
            folded = [0.0] * self.dims
            for i, v in enumerate(sk_vec):
                folded[i % self.dims] += abs(v)
            return self._combine([folded, char_vec])
        return self._combine([word_vec, char_vec])
    
    def fit_corpus(self, corpus: list[str]):
        """Fit TF-IDF on the full corpus (call after adding all blocks)."""
        if self._sklearn_available and not self._sklearn_fitted:
            self._sklearn_vec.fit(corpus)
            self._sklearn_fitted = True
    
    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        return sum(x * y for x, y in zip(a, b))
    
    @property
    def sklearn_fitted(self) -> bool:
        return self._sklearn_fitted


class ContextRetriever:
    """Vector search over context blocks."""
    
    def __init__(self, store: ContextStore, engine: EmbeddingEngine):
        self.store = store
        self.engine = engine
    
    def search(self, query: str, top_k: int = 5,
               min_similarity: float = 0.2,
               tag_filter: list[str] | None = None) -> list[dict]:
        """Return top-k most relevant context blocks."""
        query_vec = self.engine.embed(query)
        scored = []
        for block in self.store.blocks.values():
            if tag_filter:
                if not any(t in block.tags for t in tag_filter):
                    continue
            if not block.embedding:
                block.embedding = self.engine.embed(block.content)
            sim = self.engine.cosine_similarity(query_vec, block.embedding)
            # Blend: 70% similarity, 30% importance
            score = 0.7 * sim + 0.3 * block.importance
            if sim >= min_similarity:
                scored.append({
                    "id": block.id,
                    "content": block.content,
                    "source": block.source,
                    "tags": block.tags,
                    "similarity": round(sim, 4),
                    "score": round(score, 4),
                    "importance": round(block.importance, 4),
                    "age_days": round((time.time() - block.timestamp) / 86400, 1),
                })
        scored.sort(key=lambda x: x["score"], reverse=True)
        self.store.save()  # refresh decay
        return scored[:top_k]
    
    def rebuild_embeddings(self):
        """Re-embed all blocks (run after bulk import)."""
        count = 0
        for b in self.store.blocks.values():
            if not b.embedding:
                b.embedding = self.engine.embed(b.content)
                count += 1
        if count:
            self.store.save()
        return count


# Global singletons
_store: ContextStore | None = None
_retriever: ContextRetriever | None = None


def get_store(data_dir: str = "/home/ayan/context_memory/data") -> ContextStore:
    global _store
    if _store is None:
        _store = ContextStore(data_dir)
    return _store


def get_retriever(data_dir: str = "/home/ayan/context_memory/data") -> ContextRetriever:
    global _retriever
    if _retriever is None:
        _retriever = ContextRetriever(get_store(data_dir), EmbeddingEngine())
    return _retriever
