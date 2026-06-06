#!/usr/bin/env python3
"""
ctx_hermes_integration.py - Hermes context memory integration helper.

Call search_context(query, top_k=N) from any Hermes subagent/tool to
pull relevant local context and format it for the LLM prompt.

Usage from terminal:
    /home/ayan/.local/bin/ctx search "query" -k 5

Or import directly:
    from ctx_hermes_integration import build_context_prompt, ingest_memory
"""
import subprocess
import sys
import json
from pathlib import Path

CTX_BIN = "/home/ayan/.local/bin/ctx"


def search_context(query: str, top_k: int = 5, min_similarity: float = 0.2) -> str:
    """Return formatted context for LLM prompt injection."""
    try:
        result = subprocess.run(
            [CTX_BIN, "search", query, "-k", str(top_k), "--min-sim", str(min_similarity)],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"[context_memory search error: {e.stderr.strip()}]"


def ingest_memory(content: str, source: str = "hermes", tags: str = "") -> None:
    """Add a memory to the context store (side-effect)."""
    tag_args = ["-t", tags] if tags else []
    subprocess.run(
        [CTX_BIN, "add", content, "-s", source] + tag_args,
        capture_output=True
    )


def build_context_prompt(query: str, top_k: int = 5) -> str:
    """
    Build a context-rich prompt prefix for the LLM.

    Appends the retrieved context into the prompt so the model
    has access to relevant local memory without needing additional tokens
    beyond what the page-file search returns.
    """
    hits = search_context(query, top_k=top_k)
    header = "=== LOCAL CONTEXT MEMORY ===\n"
    footer = "\n=== END LOCAL CONTEXT ===\nUse the above context to answer accurately.\n"
    return header + hits + footer if hits else ""


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "default context"
    print(build_context_prompt(q))
