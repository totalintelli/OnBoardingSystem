# OnBoardingSystem

기업 내 신규 직원의 온보딩 시스템(회사 내규, 업무 규정 등)

신입 직원이 사내 규정을 질문하면 관련 규정 문서를 검색해 근거 기반으로 답변하고 출처를 표시하는 RAG(검색 증강 생성) 챗봇 프로토타입이다. 채용 과제용 프로토타입으로, 완성된 상용 서비스가 아니라 핵심 기능과 설계 판단을 보여주는 데 목적이 있다. 자세한 설계 배경은 [설계 스펙](docs/superpowers/specs/2026-08-31-onboarding-rag-chatbot-design.md), 구현 순서는 [구현 계획](docs/superpowers/plans/2026-08-31-onboarding-rag-chatbot.md) 문서를 참고한다.

## On-premise 전제

연차 일수, 경비 한도 등 사내 규정에 포함된 조직 내부 수치가 외부 LLM 제공사 서버로 전송되는 경로를 차단하기 위해, 임베딩·생성 모델을 외부 API로 호출하지 않고 전부 로컬에서 실행한다. 별도 API 토큰 없이 HuggingFace 오픈소스 모델을 로컬에 내려받아 구동한다.

## 아키텍처

```
data/docs/*.md (더미 규정 문서)
        │
        ▼
   rag.py (문서 청크 분할 → 임베딩 → 코사인 유사도 검색)
        │  top-k chunk + 유사도 점수
        ▼
   llm.py (로컬 소형 LLM, 검색 결과를 컨텍스트로 답변 생성)
        │  답변 + 참조 문서 목록
        ▼
   app.py (Streamlit 채팅 UI, 답변 아래 출처 표시)
```

## 진행 상태

- [x] 더미 사내 규정 문서 6종 (`data/docs/`)
- [x] RAG 검색 모듈 (`rag.py`) — 문서 청크 임베딩, 코사인 유사도 검색
- [ ] 로컬 LLM 답변 생성 모듈 (`llm.py`)
- [ ] Streamlit 채팅 UI (`app.py`)

## 실행 방법

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`llm.py`, `app.py` 구현 완료 후 다음 명령으로 실행한다 (아직 구현 전):

```bash
streamlit run app.py
```

## 사용 모델

- 임베딩: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- 생성: `Qwen/Qwen2.5-1.5B-Instruct`

두 모델 모두 CPU에서 동작하며 API 토큰이 필요 없다.
