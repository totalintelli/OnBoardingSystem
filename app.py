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
            results = retriever.search(query, top_k=2)
            answer = generator.generate(query, results)
        st.write(answer["answer"])
        if answer["sources"]:
            st.caption("참조 문서: " + ", ".join(answer["sources"]))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer["answer"], "sources": answer["sources"]}
    )
