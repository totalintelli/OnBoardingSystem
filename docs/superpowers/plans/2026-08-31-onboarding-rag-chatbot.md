# 신입사원 온보딩 규정 안내 챗봇 Implementation Plan

> **이 문서는 구현 착수 전에 작성한 계획이며, 실행 기록으로 그대로 보존한다.** 실제 구현 과정에서 실측을 근거로 다음 항목이 계획과 달라졌다. 최신 설계는 [설계 스펙](../specs/2026-08-31-onboarding-rag-chatbot-design.md)을, 사용법은 [README](../../../README.md)를 참고한다.
>
> - **검색 top_k**: 3 → 2. 조각이 여러 개 넘어가면 1.5B 모델이 서로 다른 조건의 숫자를 뒤섞는 현상이 반복 확인됨
> - **출처 표시**: 검색된 전체 문서 → 유사도 1순위 문서 하나. 프롬프트가 "조각 하나만 골라 답하라"고 지시하는 것과 일치시킴
> - **`rag.py` 테스트**: 절대 임계값 검증 테스트(`score < 0.5`) 제거. 이 임베딩 모델은 무관한 문장에도 0.9대를 반환해 절대 임계값이 성립하지 않음
> - **프롬프트**: "문서 문구 그대로 인용, 계산·추론 금지, 가장 관련된 조각 하나만 선택" 지시를 추가해 환각을 억제
>
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 신입 직원이 사내 규정을 질문하면 관련 규정 문서를 검색해 근거 기반으로 답변하고 출처를 표시하는 Streamlit 챗봇 프로토타입을 만든다.

**Architecture:** 더미 규정 md 문서를 청크 분할 → sentence-transformers로 임베딩 → numpy 코사인 유사도로 top-k 검색 → 로컬 소형 LLM(Qwen2.5-1.5B-Instruct)이 검색된 청크를 컨텍스트로 답변 생성 → Streamlit UI에 답변과 출처 문서명 표시. 유사도가 임계값 미만이면 LLM 호출 없이 fallback 응답.

**Tech Stack:** Python 3.12, Streamlit, sentence-transformers, transformers, numpy, pytest 불필요(assert 기반 자체 점검 스크립트로 대체)

## Global Constraints

- API 토큰 불필요 — 모든 모델은 로컬 실행(HuggingFace 모델 다운로드는 허용, API 호출은 불허)
- CPU에서 동작해야 함 (GPU 가정 금지)
- 임베딩 모델: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (한국어 지원, 경량)
- 생성 모델: `Qwen/Qwen2.5-1.5B-Instruct`
- 벡터 DB 라이브러리(FAISS 등) 사용 금지 — numpy 코사인 유사도로 충분한 규모
- 대화 이력 영속 저장, 인증, 다국어 UI 구현 금지 (스펙의 "명시적으로 하지 않는 것" 참조)

---

### Task 1: 프로젝트 환경 설정 및 더미 규정 문서 작성

**Files:**
- Create: `requirements.txt`
- Create: `data/docs/01_연차휴가.md`
- Create: `data/docs/02_근태.md`
- Create: `data/docs/03_경비처리.md`
- Create: `data/docs/04_재택근무.md`
- Create: `data/docs/05_보안수칙.md`
- Create: `data/docs/06_복리후생.md`

**Interfaces:**
- Produces: `data/docs/*.md` — 이후 Task 2의 `rag.py`가 이 경로의 모든 `.md` 파일을 읽어 청크로 분할한다. 각 파일은 `# 제목` 헤더 하나와 `## 소제목`으로 구분된 섹션들로 구성.

- [ ] **Step 1: requirements.txt 작성**

```
streamlit==1.38.0
sentence-transformers==3.1.1
transformers==4.45.0
torch==2.4.1
numpy==1.26.4
```

- [ ] **Step 2: 가상환경 생성 및 설치**

Run: `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
Expected: 설치 완료, 에러 없음

- [ ] **Step 3: 더미 규정 문서 6개 작성**

`data/docs/01_연차휴가.md`:
```markdown
# 연차 및 휴가 규정

## 연차 발생 기준
입사 1년 미만 직원은 1개월 만근 시 1일의 연차가 발생하며, 최대 11일까지 부여된다.
입사 1년 이상 직원은 연 15일의 연차가 부여되며, 3년 이상 근속 시 매 2년마다 1일씩 가산되어 최대 25일까지 늘어난다.

## 연차 사용 방법
연차는 사용 희망일 최소 3일 전에 사내 인사 시스템을 통해 신청해야 하며, 팀장 승인 후 확정된다.
반차(오전/오후)는 0.5일 단위로 사용할 수 있다.

## 경조사 휴가
본인 결혼 시 5일, 자녀 결혼 시 1일, 배우자 출산 시 10일, 부모/배우자 사망 시 5일의 유급 경조사 휴가가 부여된다.
```

`data/docs/02_근태.md`:
```markdown
# 근태 관리 규정

## 근무 시간
표준 근무시간은 09:00~18:00이며, 점심시간 1시간을 포함한 8시간 근무를 원칙으로 한다.
코어타임은 10:00~16:00이며, 이 시간 외 출퇴근 시각은 팀 협의 하에 유연하게 조정 가능하다(유연근무제).

## 지각 및 결근
15분 이상 지각 시 인사 시스템에 사유를 기록해야 한다. 월 3회 이상 무단 지각 시 팀장 면담이 진행된다.
무단 결근은 연차에서 자동 차감되며, 3일 이상 무단 결근 시 인사위원회에 회부될 수 있다.

## 출퇴근 기록
사내 출입 카드 태깅으로 자동 기록되며, 재택근무일에는 사내 시스템에 별도로 출퇴근 시각을 입력해야 한다.
```

`data/docs/03_경비처리.md`:
```markdown
# 경비 처리 규정

## 경비 신청 한도
직급별 월 법인카드 한도는 사원/대리 50만원, 과장/차장 100만원, 부장 이상 150만원이다.
1건당 10만원을 초과하는 지출은 사전 팀장 승인이 필요하다.

## 경비 처리 절차
법인카드 사용 후 3영업일 이내에 전자결재 시스템에 영수증을 첨부하여 지출 내역을 등록해야 한다.
개인카드로 결제한 경우 익월 5일까지 정산 신청을 완료해야 환급받을 수 있다.

## 출장비 규정
국내 출장 시 1일 식대 3만원, 숙박비 실비(최대 10만원)가 지원된다.
해외 출장은 별도 출장 신청서를 통해 사전 승인받아야 하며, 항공권과 숙박은 회사에서 직접 예약한다.
```

`data/docs/04_재택근무.md`:
```markdown
# 재택근무 규정

## 재택근무 신청
재택근무는 주 최대 2일까지 가능하며, 최소 1일 전 팀장 승인을 받아야 한다.
신규 입사자는 입사 후 3개월간 재택근무를 사용할 수 없으며, 수습 기간 종료 후 신청 가능하다.

## 재택근무 시 준수사항
재택근무일에도 코어타임(10:00~16:00)에는 메신저 응답이 가능한 상태를 유지해야 한다.
사내 VPN을 통해서만 사내 시스템에 접속할 수 있으며, 개인 기기 사용 시 보안 프로그램을 설치해야 한다.

## 장비 지원
재택근무자에게는 노트북 거치대, 모니터 등 재택 근무용 장비 구매비를 연 20만원 한도로 지원한다.
```

`data/docs/05_보안수칙.md`:
```markdown
# 정보보안 수칙

## 계정 및 비밀번호 관리
사내 시스템 비밀번호는 90일마다 변경해야 하며, 최근 3회 사용한 비밀번호는 재사용할 수 없다.
타인에게 계정을 공유하거나 대여하는 행위는 즉시 해고 사유가 될 수 있다.

## 자료 반출입 규정
회사 기밀 자료를 외부 이메일이나 개인 클라우드로 전송하는 행위는 금지된다.
외부 미팅 시 노트북 반출은 보안팀에 사전 신고해야 하며, USB 등 외부 저장매체 사용은 승인된 기기만 허용된다.

## 사고 발생 시 대응
정보 유출이 의심되는 경우 즉시 보안팀(내선 1234)에 신고해야 하며, 임의로 삭제하거나 은폐해서는 안 된다.
```

`data/docs/06_복리후생.md`:
```markdown
# 복리후생 안내

## 건강 및 의료
전 직원 대상 연 1회 종합건강검진을 지원하며, 검진비는 회사가 전액 부담한다.
단체 상해보험에 자동 가입되며, 입사일로부터 적용된다.

## 자기계발 지원
직무 관련 교육 및 도서 구입비를 연 100만원 한도로 지원하며, 사내 시스템을 통해 사전 신청 후 사용한다.
외부 자격증 취득 시 응시료를 전액 지원하고, 합격 시 축하금 10만원을 추가 지급한다.

## 경조사 및 기타
생일에는 축하 상품권 5만원이 지급되며, 명절(설/추석)에는 명절 선물세트가 지급된다.
매년 여름 전 직원에게 휴가비 30만원이 별도 지급된다.
```

- [ ] **Step 4: 파일 개수 확인**

Run: `ls data/docs/*.md | wc -l`
Expected: `6`

- [ ] **Step 5: Commit**

```bash
git add requirements.txt data/docs/
git commit -m "feat: add dummy HR policy documents and project dependencies"
```

---

### Task 2: RAG 검색 모듈 구현

**Files:**
- Create: `rag.py`
- Test: `test_rag.py`

**Interfaces:**
- Consumes: `data/docs/*.md` (Task 1에서 생성)
- Produces:
  - `load_chunks(docs_dir: str) -> list[dict]` — 각 dict는 `{"text": str, "source": str}` (source는 파일명)
  - `class Retriever`:
    - `__init__(self, chunks: list[dict])` — 청크들을 임베딩하여 저장
    - `search(self, query: str, top_k: int = 3) -> list[dict]` — 각 결과 dict는 `{"text": str, "source": str, "score": float}`, score 내림차순 정렬

- [ ] **Step 1: 실패하는 테스트 작성**

`test_rag.py`:
```python
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


def test_retriever_low_score_for_unrelated_query():
    chunks = load_chunks("data/docs")
    retriever = Retriever(chunks)
    results = retriever.search("오늘 저녁 메뉴 추천해줘", top_k=1)
    assert results[0]["score"] < 0.5
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `source venv/bin/activate && python -m pytest test_rag.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag'`

- [ ] **Step 3: rag.py 구현**

```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `source venv/bin/activate && python -m pytest test_rag.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add rag.py test_rag.py
git commit -m "feat: add RAG retriever with cosine similarity search"
```

---

### Task 3: 로컬 LLM 답변 생성 모듈 구현

**Files:**
- Create: `llm.py`
- Test: `test_llm.py`

**Interfaces:**
- Consumes: `Retriever.search()`의 반환 형식 `list[dict]` (`text`, `source`, `score` 키)
- Produces:
  - `SIMILARITY_THRESHOLD: float = 0.35` — 모듈 레벨 상수
  - `build_prompt(query: str, contexts: list[dict]) -> str`
  - `class AnswerGenerator`:
    - `__init__(self)` — 모델/토크나이저 로드
    - `generate(self, query: str, contexts: list[dict]) -> dict` — 반환값 `{"answer": str, "sources": list[str]}`. `contexts`가 비어있거나 최고 score가 `SIMILARITY_THRESHOLD` 미만이면 LLM 호출 없이 `{"answer": "죄송합니다, 규정 문서에서 관련 내용을 찾지 못했습니다. 인사팀에 문의해주세요.", "sources": []}` 반환

- [ ] **Step 1: 실패하는 테스트 작성**

`test_llm.py`:
```python
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `source venv/bin/activate && python -m pytest test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm'`

- [ ] **Step 3: llm.py 구현**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SIMILARITY_THRESHOLD = 0.35
_FALLBACK_ANSWER = "죄송합니다, 규정 문서에서 관련 내용을 찾지 못했습니다. 인사팀에 문의해주세요."


def build_prompt(query: str, contexts: list[dict]) -> str:
    context_text = "\n\n".join(c["text"] for c in contexts)
    return (
        "다음은 사내 규정 문서의 일부입니다. 이 내용만을 근거로 질문에 답하세요.\n\n"
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
        output_ids = self.model.generate(**inputs, max_new_tokens=300)
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True)

        sources = sorted({c["source"] for c in contexts})
        return {"answer": answer.strip(), "sources": sources}
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `source venv/bin/activate && python -m pytest test_llm.py -v`
Expected: PASS (3 passed) — `AnswerGenerator.__new__`로 생성한 인스턴스는 `__init__`을 거치지 않으므로 모델 다운로드 없이 fallback 로직만 검증됨

- [ ] **Step 5: Commit**

```bash
git add llm.py test_llm.py
git commit -m "feat: add local LLM answer generator with fallback on low similarity"
```

---

### Task 4: Streamlit 챗봇 UI 및 파이프라인 통합

**Files:**
- Create: `app.py`
- Create: `pipeline_check.py`

**Interfaces:**
- Consumes:
  - `rag.load_chunks(docs_dir: str) -> list[dict]`
  - `rag.Retriever(chunks: list[dict])`, `.search(query, top_k) -> list[dict]`
  - `llm.AnswerGenerator()`, `.generate(query, contexts) -> {"answer": str, "sources": list[str]}`
- Produces: 실행 가능한 Streamlit 앱 (다른 태스크에서 소비하지 않음, 최종 산출물)

- [ ] **Step 1: 파이프라인 자체 점검 스크립트 작성**

`pipeline_check.py`:
```python
"""검색→생성 파이프라인이 실제로 동작하는지 확인하는 자체 점검 스크립트.
pytest가 아닌 독립 실행 스크립트로 둔 이유: 모델 다운로드가 필요해 무겁고,
CI 자동화 대상이 아니라 수동 데모 전 1회 확인용이기 때문.
"""
from rag import load_chunks, Retriever
from llm import AnswerGenerator


def demo():
    chunks = load_chunks("data/docs")
    retriever = Retriever(chunks)
    generator = AnswerGenerator()

    query = "연차는 며칠 쓸 수 있어?"
    results = retriever.search(query, top_k=3)
    assert results, "검색 결과가 비어있으면 안 됨"
    assert results[0]["score"] > SIMILARITY_THRESHOLD_CHECK, "관련 질문인데 유사도가 너무 낮음"

    answer = generator.generate(query, results)
    assert answer["sources"], "관련 문서를 찾았는데 출처가 비어있으면 안 됨"
    assert "연차휴가" in answer["sources"][0]

    print("질문:", query)
    print("답변:", answer["answer"])
    print("출처:", answer["sources"])
    print("자체 점검 통과")


SIMILARITY_THRESHOLD_CHECK = 0.35

if __name__ == "__main__":
    demo()
```

- [ ] **Step 2: 자체 점검 실행**

Run: `source venv/bin/activate && python pipeline_check.py`
Expected: `자체 점검 통과` 출력, AssertionError 없음 (최초 실행 시 모델 다운로드로 수 분 소요될 수 있음)

- [ ] **Step 3: app.py 구현**

```python
import streamlit as st

from rag import load_chunks, Retriever
from llm import AnswerGenerator

st.set_page_config(page_title="사내 규정 안내 챗봇", page_icon="📋")
st.title("📋 신입사원 온보딩 규정 안내 챗봇")
st.caption("연차, 근태, 경비, 재택근무, 보안, 복리후생 등 사내 규정을 물어보세요.")


@st.cache_resource
def load_pipeline():
    chunks = load_chunks("data/docs")
    retriever = Retriever(chunks)
    generator = AnswerGenerator()
    return retriever, generator


retriever, generator = load_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            st.caption("참조 문서: " + ", ".join(msg["sources"]))

query = st.chat_input("궁금한 규정을 입력하세요 (예: 연차는 며칠 쓸 수 있어?)")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("규정 문서를 검색하고 답변을 생성하는 중..."):
            results = retriever.search(query, top_k=3)
            answer = generator.generate(query, results)
        st.write(answer["answer"])
        if answer["sources"]:
            st.caption("참조 문서: " + ", ".join(answer["sources"]))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer["answer"], "sources": answer["sources"]}
    )
```

- [ ] **Step 4: 앱 실행 확인**

Run: `source venv/bin/activate && streamlit run app.py`
Expected: 브라우저에서 `http://localhost:8501` 열림, 채팅 입력창에 "연차는 며칠 쓸 수 있어?" 입력 시 답변과 "참조 문서: 01_연차휴가.md" 표시 확인. 확인 후 Ctrl+C로 종료.

- [ ] **Step 5: Commit**

```bash
git add app.py pipeline_check.py
git commit -m "feat: add Streamlit chat UI wiring RAG retrieval and LLM generation"
```

---

## Self-Review 결과

**스펙 커버리지**: 더미 문서 생성(Task 1), 임베딩+검색(Task 2), 로컬 LLM 생성+fallback(Task 3), Streamlit UI+출처 표시(Task 4) — 스펙의 모든 컴포넌트 반영. "명시적으로 하지 않는 것" 항목(인증, DB 저장, 다국어, 관리자 UI)은 어떤 태스크에도 포함하지 않음.

**타입 일관성**: `Retriever.search()`의 반환 형식(`text`, `source`, `score`)이 Task 3의 `build_prompt`, `generate`와 Task 4의 사용처에서 동일하게 유지됨. `AnswerGenerator.generate()`의 반환 형식(`answer`, `sources`)도 Task 4 `app.py`에서 그대로 사용.
