import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA


def generate_response(uploaded_file, openai_api_key, query_text):
    #Read uploaded text file
    document_text = uploaded_file.read().decode("utf-8")

    # Split document into smaller chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )
    documents = text_splitter.create_documents([document_text])

    # Create embeddings(doc chunks)
    embeddings = OpenAIEmbeddings(
        openai_api_key=openai_api_key,
        model="text-embedding-3-small")
    #Store embeddings in Chroma vectorStore
    vector_db = Chroma.from_documents(documents, embeddings)
    retriever = vector_db.as_retriever()
    # Create the question-answering chain
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model="gpt-4o-mini",
        temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever)

    result = qa_chain.invoke({"query": query_text})
    return result["result"]


st.set_page_config(page_title="Ask the Doc App")
st.title("Ask the Doc App")

st.write(
    "Upload a text file and ask a question about it. "
    "The app uses LangChain, embeddings, Chroma, and a question-answering chain."
)

uploaded_file = st.file_uploader("Upload a text file", type="txt")

query_text = st.text_input(
    "Enter your question:",
    placeholder="Example: What is this document mainly about?",
    disabled=not uploaded_file)

with st.form("question_form", clear_on_submit=False):
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        disabled=not (uploaded_file and query_text))

    submitted = st.form_submit_button(
        "Submit",
        disabled=not (uploaded_file and query_text))

    if submitted:
        if not openai_api_key.startswith("sk-"):
            st.error("Please enter a valid OpenAI API key.")
        else:
            with st.spinner("Generating answer..."):
                answer = generate_response(uploaded_file, openai_api_key, query_text)
                st.info(answer)
            del openai_api_key