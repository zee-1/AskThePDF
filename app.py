import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores.chroma import Chroma
import os

# Load environment variables (if using API keys)
text_splitter = RecursiveCharacterTextSplitter()
embeddings = OpenAIEmbeddings(openai_api_key= os.environ["OPENAI_API_KEY"])
vectorstore = Chroma(embedding_function=embeddings)
llm = ChatOpenAI(temperature=0.7)  # Adjust temperature as needed

# Streamlit UI elements
st.title("Ask the PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf",)

if uploaded_file:
    with open("./data.pdf","wb+") as f:
        f.write(uploaded_file.getvalue())
        f.close()
    pdf_loader = PyPDFLoader("./data.pdf")
    chunks = pdf_loader.load_and_split()
    vectorstore.add_documents(chunks)
    chain = RetrievalQA.from_chain_type(llm=llm,retriever=vectorstore.as_retriever())

    query = st.chat_input("Ask a question about the PDF")
    if query:
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("ai"):
            response = chain(query)
            st.write(response["result"])
