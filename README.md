# Local Context Memory for LLMs

A persistent, SSD-backed "page file" for LLM context. Local embeddings + vector search + JSON storage. Works as a Hermes skill, or standalone.

```
/home/ayan/context_memory/
├── venv/bin/python3
├── src/ctx_memory/
│   ├── __init__.py
│   ├── core.py                # ContextStore, EmbeddingEngine, ContextRetriever
│   └── cli.py                 # ctx CLI
├── data/
│   └── index.json             # Persisted block store (SSD-backed)
├── ctx_hermes_integration.py  # Hermes helper: build_context_prompt()
└── bin/ctx -> venv/bin/python3 cli.py
```

## Use case

Stop re-prompting. Store durable facts (client IDs, system IPs, prefs, project paths, guard rules) once, then pull them by semantic relevance. Hermes uses local context first, falls back to flash knowledge only when local context is not available.

## Install

```bash
git clone https://github.com/real-linuxwala/local_memory_context_LLMs.git
cd local_memory_context_LLMs
python3 -m venv venv
./venv/bin/python3 -m pip install --quiet numpy scikit-learn
mkdir -p /home/ayan/.local/bin
cat > /home/ayan/.local/bin/ctx <<'EOF'
#!/usr/bin/env bash
VENV="$HOME/local_memory_context_LLMs/venv/bin/python3"
CLI="$HOME/local_memory_context_LLMs/src/ctx_memory/cli.py"
exec "$VENV" "$CLI" "$@"
EOF
chmod +x /home/ayan/.local/bin/ctx
```

## CLI

```text
ctx add   "content" [-s SOURCE] [-t tag1,tag2] [-i IMPORTANCE]
ctx search "query" [-k TOP_K]
ctx get <id>
ctx list
ctx stats
ctx delete <id>
ctx rebuild    # fit TF-IDF + re-embed stored blocks
```

## Data model

| Field | Meaning |
|---|---|
| id | md5(content[:12]) |
| content | stored string |
| source | provenance label |
| tags | searchable labels |
| timestamp | unix time |
| access_count | popularity |
| importance | 0..1, decays weekly |
| embedding | 256-dim stable MD5-hashed TF-IDF + char 3-gram |

## How it works

- **256-dimensional vectors** are stable across process restarts (MD5-hashed features, not Python's non-deterministic `hash()`).
- **Two feature streams**: TF-IDF vocabulary (scikit-learn) + character 3-gram fallback (stdlib only).
- **Similarity**: cosine on L2-normalized vectors.
- **Score**: `0.7 * similarity + 0.3 * importance`.
- **Storage**: plain JSON at `data/index.json`.

## Hermes integration

Rule: **use local context whenever possible. Only use flash context when local context is not available.**

```bash
/home/ayan/.local/bin/ctx search "<query>" -k 5 --min-sim 0.2
```

If results exist, answer from them. If empty, fall back to model knowledge.

See [local-context-memory skill](/home/ayan/.hermes/skills/productivity/local-context-memory/SKILL.md) for the canonical Hermes workflow.

## Verified surface

- Linux (Ubuntu), Python 3.14 venv
- Persistence confirmed across fresh CLI invocations
- Degrade gracefully if numpy/scikit-learn missing (character 3-gram fallback)

## License

MIT
