import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain

st.title("INsert your pdf")

uploaded_file=st.file_uploader("upload your pdf",type=["pdf"])
if uploaded_file is not None:
    
    with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path=temp_file.name

    loader=PyPDFLoader(temp_file_path)
    docs=loader.load()
    st.write("Loaded the document successfully")

    os.remove(temp_file_path)

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    documents=text_splitter.split_documents(docs)
    st.write("Chunked the document successfully")

    embeddings=OllamaEmbeddings(model="llama3.2:1b")
    # db=Chroma.from_documents(documents, embedding=embeddings)
    db=FAISS.from_documents(documents[:10],embedding=embeddings)
    st.write("embedded the document successfully")

    llm=Ollama(model="llama3.2:1b")
    prompt=ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
think step by step before providing a detailed answer.
Context: {context}
Question: {input}
""")

    document_chain=create_stuff_documents_chain(llm,prompt)

    retriever=db.as_retriever()

    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    st.write("Chains created successfully")

    query=st.text_input("Ask a question about the PDF")
    if query:
        response=retrieval_chain.invoke({"input":query})
        st.write(response["answer"])
