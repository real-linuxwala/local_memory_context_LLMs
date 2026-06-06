#!/usr/bin/env python3
"""
CLI tool: ctx
Usage:
    ctx add "content" [-s SOURCE] [-t TAG1,TAG2] [-i IMPORTANCE]
    ctx search "query" [-k TOP_K]
    ctx get <id>
    ctx delete <id>
    ctx list
    ctx stats
    ctx rebuild  # re-embed all blocks
"""
import sys
import datetime
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import get_store, get_retriever


def time_ctime(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def cmd_add(args):
    store = get_store()
    tags = args.tags.split(",") if args.tags else []
    imp = args.importance if args.importance else 1.0
    block_id = store.add(args.content, source=args.source or "manual", tags=tags, importance=imp)
    print(f"stored {block_id}  ({len(args.content)} chars)")
    if args.source:
        print(f"  source: {args.source}")
    if tags:
        print(f"  tags: {', '.join(tags)}")
    # Embed after add so it can be searched immediately
    retriever = get_retriever()
    b = store.blocks[block_id]
    b.embedding = retriever.engine.embed(b.content)
    store.save()


def cmd_search(args):
    retriever = get_retriever()
    results = retriever.search(args.query, top_k=args.k, min_similarity=args.min_sim)
    if not results:
        print("No matching context found.")
        return
    print(f"found {len(results)} matches for \"{args.query}\"\n")
    for r in results:
        print(f"[{r['id']}]  sim={r['similarity']}  score={r['score']}  age={r['age_days']}d  src={r['source']}")
        print(f"  tags: {', '.join(r['tags'])}")
        # Truncate long content
        content = r['content']
        if len(content) > 300:
            content = content[:297] + "..."
        for line in content.split("\n"):
            print(f"  {line}")
        print()


def cmd_get(args):
    store = get_store()
    b = store.get(args.id)
    if not b:
        print(f"not found: {args.id}")
        sys.exit(1)
    print(f"id:      {b.id}")
    print(f"source:  {b.source}")
    print(f"tags:    {', '.join(b.tags)}")
    print(f"stored:  {time_ctime(b.timestamp)}")
    print(f"hits:    {b.access_count}")
    print(f"len:     {len(b.content)} chars")
    print()
    print(b.content)
def cmd_delete(args):
    store = get_store()
    if store.delete(args.id):
        print(f"deleted {args.id}")
    else:
        print(f"not found: {args.id}")
        sys.exit(1)


def cmd_list(args):
    store = get_store()
    items = sorted(store.blocks.values(), key=lambda b: b.access_count, reverse=True)
    fmt = "{:<14} {:<30} {:<14} {:<6} {:<8}"
    print(fmt.format("ID", "SOURCE", "TAGS", "HITS", "SIZE"))
    print("-" * 80)
    for b in items[:args.limit]:
        tags_str = ", ".join(b.tags[:2]) or "-"
        if len(tags_str) > 13:
            tags_str = tags_str[:11] + ".."
        src = b.source[:29] if b.source else "-"
        print(fmt.format(b.id[:13], src, tags_str, b.access_count, f"{len(b.content)}c"))


def cmd_stats(args):
    store = get_store()
    s = store.stats()
    print(f"total blocks : {s['total_blocks']}")
    print(f"storage path : {s['storage_path']}")
    for src, cnt in sorted(s['sources'].items(), key=lambda x: -x[1]):
        print(f"  {src}: {cnt}")


def cmd_rebuild(_):
    retriever = get_retriever()
    corupus = [b.content for b in retriever.store.blocks.values()]
    retriever.engine.fit_corpus(corupus)
    done = retriever.rebuild_embeddings()
    mode = "TF-IDF + n-gram" if retriever.engine.sklearn_fitted else "n-gram only"
    print(f"re-embedded {done} blocks ({mode})")


def main():
    p = argparse.ArgumentParser(description="ctx - Hermes local context memory")
    sub = p.add_subparsers(dest="cmd")

    add_p = sub.add_parser("add")
    add_p.add_argument("content")
    add_p.add_argument("-s", "--source", default="")
    add_p.add_argument("-t", "--tags", default="")
    add_p.add_argument("-i", "--importance", type=float, default=1.0)

    search_p = sub.add_parser("search")
    search_p.add_argument("query")
    search_p.add_argument("-k", type=int, default=5)
    search_p.add_argument("--min-sim", type=float, default=0.2)

    sub.add_parser("stats")
    sub.add_parser("rebuild")
    
    get_p = sub.add_parser("get")
    get_p.add_argument("id")

    del_p = sub.add_parser("delete")
    del_p.add_argument("id")

    list_p = sub.add_parser("list")
    list_p.add_argument("-n", "--limit", type=int, default=30)

    args = p.parse_args()
    if not args.cmd:
        p.print_help()
        sys.exit(1)

    cmds = {
        "add": cmd_add, "search": cmd_search, "get": cmd_get,
        "delete": cmd_delete, "list": cmd_list, "stats": cmd_stats, "rebuild": cmd_rebuild,
    }
    try:
        cmds[args.cmd](args)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
