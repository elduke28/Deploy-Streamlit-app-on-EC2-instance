import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import time

def generate_response(uploaded_file, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.create_documents(documents)
        # Select embeddings
        ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text",base_url="http://100.118.115.122:11434",temperature="0.1")
        # Create a vectorstore from documents
        db = Chroma.from_documents(documents=texts,
                                   embedding=ollama_embeddings,
                                   collection_name="rag-url",
                                   persist_directory="data1")
        # Save the Chroma database to disk
        db.persist()
        # Create retriever interface
        retriever = db.as_retriever(search_kwargs={"k": 3})
        # Create QA chain
        llm = Ollama(model="hestia",base_url="http://100.118.115.122:11434",keep_alive="-1",temperature="0.8")
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
        return qa.invoke(query_text)

def stream_data(text, delay: float=0.05):
    for word in text.split():
        yield word + " "
        time.sleep(delay)
# Page title
st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')

# File upload
uploaded_file = st.file_uploader('Upload an article', type='txt')
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted:
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, query_text)
            result = response["result"]

if len(result):
    st.write_stream(stream_data(result))