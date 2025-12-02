# ask-the-doc-app-ai
Streamlit + LangChain "Ask the Doc" app(upload a text file and ask the questions about it)

A simple Streamlit + LangChain app for question answering over documents.
- Upload a `.txt` document (e.g. `state_of_the_union.txt`)
- You can ask a natural-language question about the document.
- The app splits the text into chunks, creates OpenAI embeddings, stores them in a Chroma vector store, and uses a RetrievalQA chain to answer questions based only on the uploaded file

 #run locally
- pip install -r requirements.txt
streamlit run ask_the_doc_app.py
