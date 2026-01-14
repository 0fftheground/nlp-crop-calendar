import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
from langchain_openai import OpenAIEmbeddings

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.infra.config import get_config


def _infer_collection(default: str = "varieties") -> str:
    raw_map = os.getenv("QDRANT_COLLECTIONS", "")
    if not raw_map:
        return default
    try:
        mapping = json.loads(raw_map)
    except json.JSONDecodeError:
        return default
    if isinstance(mapping, dict):
        return str(mapping.get("variety") or default)
    return default


def _load_variety_items(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        items = payload
    else:
        items = payload.get("varieties", [])
    normalized = []
    for item in items:
        if isinstance(item, str):
            name = item.strip()
            if name:
                normalized.append({"name": name})
            continue
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            if name:
                normalized.append({**item, "name": name})
    return normalized


def _build_embeddings(
    names: List[str],
    *,
    model: str,
) -> Tuple[List[List[float]], int]:
    cfg = get_config()
    api_key = cfg.extractor_api_key or cfg.openai_api_key
    if not api_key:
        raise ValueError("Missing OpenAI API key for embeddings.")
    base_url = cfg.extractor_api_base or cfg.openai_api_base
    kwargs = {"api_key": api_key, "model": model}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAIEmbeddings(**kwargs)
    vectors = client.embed_documents(names)
    dim = len(vectors[0]) if vectors else 0
    return vectors, dim


def _collection_exists(client: httpx.Client, url: str, name: str) -> bool:
    resp = client.get(f"{url}/collections/{name}")
    return resp.status_code == 200


def _ensure_collection(
    client: httpx.Client,
    url: str,
    name: str,
    dim: int,
    *,
    recreate: bool = False,
) -> None:
    if recreate and _collection_exists(client, url, name):
        client.delete(f"{url}/collections/{name}")
    if not _collection_exists(client, url, name):
        payload = {"vectors": {"size": dim, "distance": "Cosine"}}
        resp = client.put(f"{url}/collections/{name}", json=payload)
        resp.raise_for_status()


def _upsert_points(
    client: httpx.Client,
    url: str,
    name: str,
    items: List[Dict[str, Any]],
    vectors: List[List[float]],
    *,
    start_id: int = 1,
    batch_size: int = 64,
) -> None:
    total = len(items)
    for offset in range(0, total, batch_size):
        batch_items = items[offset : offset + batch_size]
        batch_vectors = vectors[offset : offset + batch_size]
        points = []
        for idx, (item, vector) in enumerate(zip(batch_items, batch_vectors)):
            points.append(
                {
                    "id": start_id + offset + idx,
                    "vector": vector,
                    "payload": item,
                }
            )
        resp = client.put(
            f"{url}/collections/{name}/points",
            params={"wait": "true"},
            json={"points": points},
        )
        resp.raise_for_status()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Qdrant index for varieties.")
    parser.add_argument(
        "--input",
        default=str(ROOT / "resources" / "varieties.json"),
        help="Path to varieties JSON file.",
    )
    parser.add_argument(
        "--collection",
        default=_infer_collection(),
        help="Qdrant collection name.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant HTTP URL.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        help="Embedding model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Upsert batch size.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate collection if it exists.",
    )
    args = parser.parse_args()

    items = _load_variety_items(Path(args.input))
    if not items:
        raise SystemExit("No varieties found.")

    names = [item["name"] for item in items]
    vectors, dim = _build_embeddings(names, model=args.model)
    if dim <= 0:
        raise SystemExit("Embedding dimension is invalid.")

    with httpx.Client(timeout=60, trust_env=False) as client:
        _ensure_collection(
            client,
            args.qdrant_url,
            args.collection,
            dim,
            recreate=args.recreate,
        )
        _upsert_points(
            client,
            args.qdrant_url,
            args.collection,
            items,
            vectors,
            batch_size=args.batch_size,
        )

    print(
        f"Upserted {len(items)} varieties into '{args.collection}' at {args.qdrant_url}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
