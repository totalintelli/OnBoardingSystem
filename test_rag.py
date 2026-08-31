from rag import load_chunks, Retriever


def test_load_chunks_splits_by_section():
    chunks = load_chunks("data/docs")
    assert len(chunks) > 6  # 문서 6개, 섹션 여러 개이므로 청크는 6개 초과
    assert all("text" in c and "source" in c for c in chunks)
    assert any("01_연차휴가.md" in c["source"] for c in chunks)


def test_retriever_finds_relevant_chunk():
    chunks = load_chunks("data/docs")
    retriever = Retriever(chunks)
    results = retriever.search("연차는 며칠 쓸 수 있어?", top_k=3)
    assert len(results) == 3
    assert results[0]["score"] >= results[1]["score"] >= results[2]["score"]
    assert any("연차휴가" in r["source"] for r in results)
