import glob
import os
import re

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def load_chunks(docs_dir: str) -> list[dict]:
    chunks = []
    for path in sorted(glob.glob(os.path.join(docs_dir, "*.md"))):
        source = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        sections = re.split(r"\n(?=## )", content)
        for section in sections:
            text = section.strip()
            if text:
                chunks.append({"text": text, "source": source})
    return chunks


class Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.model = SentenceTransformer(_MODEL_NAME)
        texts = [c["text"] for c in chunks]
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        query_vec = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ query_vec
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {**self.chunks[i], "score": float(scores[i])}
            for i in top_indices
        ]
