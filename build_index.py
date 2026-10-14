import sys
from pathlib import Path

import database

CHROMA_DIR = Path("chroma_store")
COLLECTION_NAME = "cv_sections"
MODEL_NAME = "all-MiniLM-L6-v2"     # small and fast; max input is about 256 tokens
MAX_WORDS_PER_CHUNK = 120           # keep chunks safely under the model's limit


# ---------- Chunking ----------

def chunk_section(text: str, max_words: int = MAX_WORDS_PER_CHUNK) -> list[str]:
    """
    Split a section into chunks of at most ~max_words, breaking only between lines
    (so a bullet point is never cut in half). Short sections stay as one chunk.
    """
    chunks, current, count = [], [], 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        n = len(line.split())
        if current and count + n > max_words:
            chunks.append("\n".join(current))
            current, count = [], 0
        current.append(line)
        count += n
    if current:
        chunks.append("\n".join(current))
    return chunks


def build_chunks(rows: list[dict]) -> list[dict]:
    """rows from the database -> list of chunk records ready to embed."""
    records = []
    for row in rows:
        for i, chunk in enumerate(chunk_section(row["text"])):
            records.append({
                "id": f"{row['cv_id']}::{row['section_name']}::{i}",
                "cv_id": row["cv_id"],
                "section": row["section_name"],
                "chunk_index": i,
                "text": chunk,
                # The section name is added only for embedding, so the model knows the context
                "embed_text": f"{row['section_name']}: {chunk}",
            })
    return records


# ---------- Model and ChromaDB ----------

_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer   # imported late: slow to load
        print(f"Loading model {MODEL_NAME} ...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    vectors = get_model().encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,     # unit length vectors: cosine similarity = dot product
        show_progress_bar=len(texts) > 20,
    )
    return vectors.tolist()


def get_collection(reset: bool = False):
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass                       # nothing to delete yet
    # Chroma defaults to L2 distance. We want cosine, so we say so explicitly.
    try:
        return client.get_or_create_collection(
            name=COLLECTION_NAME, configuration={"hnsw": {"space": "cosine"}})
    except TypeError:                  # older chromadb versions use metadata instead
        return client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


# ---------- Commands ----------

def indexed_cv_ids(collection) -> set[str]:
    data = collection.get(include=["metadatas"])
    return {m["cv_id"] for m in data["metadatas"]}


def build_index(rebuild: bool = False) -> None:
    rows = database.get_sections_for_matching()
    if not rows:
        print("No CV sections found in the database. Run database.py first.")
        sys.exit(1)

    collection = get_collection(reset=rebuild)
    already = set() if rebuild else indexed_cv_ids(collection)

    new_rows = [r for r in rows if r["cv_id"] not in already]
    if not new_rows:
        print(f"Nothing new to index ({len(already)} candidates already indexed).")
        return

    records = build_chunks(new_rows)
    print(f"Embedding {len(records)} chunks from {len({r['cv_id'] for r in records})} candidates ...")
    vectors = embed([r["embed_text"] for r in records])

    collection.upsert(                 # upsert = insert, or replace if the id exists
        ids=[r["id"] for r in records],
        embeddings=vectors,
        documents=[r["text"] for r in records],
        metadatas=[{"cv_id": r["cv_id"], "section": r["section"],
                    "chunk_index": r["chunk_index"]} for r in records],
    )
    print(f"Done. Index now holds {collection.count()} chunks.")


def show_stats() -> None:
    collection = get_collection()
    ids = indexed_cv_ids(collection)
    print(f"{collection.count()} chunks from {len(ids)} candidates")


def delete_from_index(cv_id: str) -> None:
    get_collection().delete(where={"cv_id": cv_id})
    print(f"Removed {cv_id} from the index.")


def test_search(query: str, n: int = 5) -> None:
    collection = get_collection()
    result = collection.query(query_embeddings=embed([query]), n_results=n)
    print(f'\nTop {n} chunks for: "{query}"\n')
    for cid, meta, doc, dist in zip(result["ids"][0], result["metadatas"][0],
                                    result["documents"][0], result["distances"][0]):
        similarity = 1 - dist          # Chroma returns a distance; for cosine, similarity = 1 - distance
        preview = doc.replace("\n", " ")[:90]
        print(f"{similarity:.3f}  {meta['cv_id']:28} {meta['section']:14} {preview}")


def main():
    args = sys.argv[1:]
    if "--stats" in args:
        show_stats()
    elif "--delete" in args:
        delete_from_index(args[args.index("--delete") + 1])
    elif "--test" in args:
        test_search(args[args.index("--test") + 1])
    else:
        build_index(rebuild="--rebuild" in args)


if __name__ == "__main__":
    main()
