"""검색→생성 파이프라인이 실제로 동작하는지 확인하는 자체 점검 스크립트.
pytest가 아닌 독립 실행 스크립트로 둔 이유: 모델 다운로드가 필요해 무겁고,
CI 자동화 대상이 아니라 수동 데모 전 1회 확인용이기 때문.
"""
from rag import load_chunks, Retriever
from llm import AnswerGenerator, SIMILARITY_THRESHOLD


def demo():
    chunks = load_chunks("data/docs")
    retriever = Retriever(chunks)
    generator = AnswerGenerator()

    query = "연차는 며칠 쓸 수 있어?"
    results = retriever.search(query, top_k=2)
    assert results, "검색 결과가 비어있으면 안 됨"
    assert results[0]["score"] > SIMILARITY_THRESHOLD, "관련 질문인데 유사도가 임계값보다 낮음"

    answer = generator.generate(query, results)
    assert answer["sources"], "관련 문서를 찾았는데 출처가 비어있으면 안 됨"
    assert "연차휴가" in answer["sources"][0]

    print("질문:", query)
    print("답변:", answer["answer"])
    print("출처:", answer["sources"])
    print("자체 점검 통과")


if __name__ == "__main__":
    demo()
