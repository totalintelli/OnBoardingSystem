from llm import build_prompt, AnswerGenerator, SIMILARITY_THRESHOLD


def test_build_prompt_includes_context_and_query():
    contexts = [{"text": "연차는 15일이다.", "source": "01_연차휴가.md", "score": 0.8}]
    prompt = build_prompt("연차는 며칠이야?", contexts)
    assert "연차는 15일이다." in prompt
    assert "연차는 며칠이야?" in prompt


def test_generate_falls_back_when_score_below_threshold():
    generator = AnswerGenerator.__new__(AnswerGenerator)  # 모델 로드 스킵
    contexts = [{"text": "무관한 내용", "source": "x.md", "score": SIMILARITY_THRESHOLD - 0.1}]
    result = generator.generate("아무 질문", contexts)
    assert result["sources"] == []
    assert "인사팀" in result["answer"]


def test_generate_falls_back_when_no_contexts():
    generator = AnswerGenerator.__new__(AnswerGenerator)
    result = generator.generate("아무 질문", [])
    assert result["sources"] == []
