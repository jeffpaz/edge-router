"""Full rebuild of the PazBrain search index from the wiki files.

    docker exec edge-router python -m pazbrain.reindex
"""

from pazbrain import store


def main() -> None:
    count = store.rebuild_index()
    print(f"reindexed {count} pages from {store.WIKI_DIR} into {store.INDEX_DB}")


if __name__ == "__main__":
    main()
