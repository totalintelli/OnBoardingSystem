from transformers import AutoModelForCausalLM, AutoTokenizer

_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SIMILARITY_THRESHOLD = 0.35
_FALLBACK_ANSWER = "죄송합니다, 규정 문서에서 관련 내용을 찾지 못했습니다. 인사팀에 문의해주세요."


def build_prompt(query: str, contexts: list[dict]) -> str:
    context_text = "\n\n".join(c["text"] for c in contexts)
    return (
        "다음은 사내 규정 문서의 일부입니다. 이 내용만을 근거로 질문에 답하세요.\n"
        "문서에 명시된 문구를 그대로 인용하고, 문서에 없는 계산이나 추론을 하지 마세요.\n"
        "여러 조각이 주어지면 그 중 질문과 가장 직접 관련된 조각 하나만 골라 답하고,"
        " 나머지 조각의 숫자나 조건은 답변에 섞지 마세요.\n\n"
        f"[규정 문서]\n{context_text}\n\n"
        f"[질문]\n{query}\n\n"
        "[답변]"
    )


class AnswerGenerator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(_MODEL_NAME)

    def generate(self, query: str, contexts: list[dict]) -> dict:
        if not contexts or contexts[0]["score"] < SIMILARITY_THRESHOLD:
            return {"answer": _FALLBACK_ANSWER, "sources": []}

        prompt = build_prompt(query, contexts)
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True)

        # 프롬프트가 "가장 관련된 조각 하나만 골라 답하라"고 지시하므로,
        # 출처도 유사도 1순위 조각의 문서만 표시한다.
        return {"answer": answer.strip(), "sources": [contexts[0]["source"]]}
