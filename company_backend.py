#pip install flask-sqlalchemy

#pip install pypdf

#pip install faiss-cpu

import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS 
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms.bedrock import Bedrock

def company_pdf():
    data_load = PyPDFLoader("https://www.upl-ltd.com/images/people/downloads/Leave-Policy-India.pdf")
    pdf_split = RecursiveCharacterTextSplitter(separators= ["\n\n", "\n"," ",""], chunk_size = 100, chunk_overlap=10)
    pdf_embedding = BedrockEmbeddings(
        credentials_profile_name = 'default',
        model_id = 'amazon.titan-embed-text-v1'
    )
    pdf_index = VectorstoreIndexCreator(text_splitter= pdf_split, 
                                        embedding= pdf_embedding, 
                                        vectorstore_cls= FAISS)
    db_index = pdf_index.from_loaders([data_load])
    return db_index

def company_llm():
    llm = Bedrock(
        credentials_profile_name= 'default',
        model_id= 'anthropic.claude-v2',
        model_kwargs= {
            "max_tokens_to_sample": 5000,
            "temperature": 0.1,
            "top_p": 0.8})
    return llm

def company_rag_response(index, question):
    rag_llm = company_llm()
    company_rag_query = index.query(question = question, llm = rag_llm)
    return company_rag_query
