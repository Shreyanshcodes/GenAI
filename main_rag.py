# import os
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceBgeEmbeddings
# from langchain.document_loaders import DirectoryLoader
# from langchain.document_loaders import PyPDFLoader
# from flask import Flask, request, jsonify, render_template
# import os
# import json
# from langchain import PromptTemplate, LLMChain
# from langchain.llms import CTransformers
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
# from langchain.chains import RetrievalQA
# from langchain.embeddings import HuggingFaceBgeEmbeddings
# from langchain.document_loaders import PyPDFLoader
# import os


# def initialize_store():

#     stores_folder = "stores"
#     pet_cosine_folder = "pet_cosine"

#     stores_path = os.path.join(os.getcwd(), stores_folder)
#     pet_cosine_path = os.path.join(stores_path, pet_cosine_folder)

#     if os.path.exists(pet_cosine_path) and os.path.isdir(pet_cosine_path):
#         print(f"The subfolder '{pet_cosine_folder}' exists inside '{stores_folder}'.")
#         print("Proceeding for RAG...")

#     else:
#         print(f"The subfolder '{pet_cosine_folder}' does not exist inside '{stores_folder}'.")
#         print("Proceeding for Vector DB Creation...")
#         model_name = "BAAI/bge-large-en"
#         model_kwargs = {'device': 'cpu'}
#         encode_kwargs = {'normalize_embeddings': False}
#         embeddings = HuggingFaceBgeEmbeddings(
#             model_name=model_name,
#             model_kwargs=model_kwargs,
#             encode_kwargs=encode_kwargs
#         )
#         loader = DirectoryLoader('data/', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
#         documents = loader.load()
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
#         texts = text_splitter.split_documents(documents)

#         vector_store = Chroma.from_documents(texts, embeddings, collection_metadata={"hnsw:space": "cosine"}, persist_directory="stores/pet_cosine")

#         print("Vector Store Created.......")

# def initialize_model():
#     local_llm = "tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
#     config = {
#         'max_new_tokens':1024,
#         'repetition_penalty':1.1,
#         'temperature':0.1,
#         'top_k':50,
#         'top_p':0.9,
#         'stream':True,
#         'threads':int(os.cpu_count()/2)

#     }
#     global llm
#     llm = CTransformers(
#         model=local_llm,
#         model_type="mistral",
#         lib="avx2",
#         **config
#     )

#     print("LLM Initialized....")

#     prompt_template = """Use the following pieces of information to answer the user's question.
#     If you don't know the answer, just say that you don't know, don't try to make up an answer.

#     Context: {context}
#     Question: {question}

#     Only return the helpful answer below and nothing else.
#     Helpful answer:
#     """

#     model_name = "BAAI/bge-large-en"
#     model_kwargs = {'device': 'cpu'}
#     encode_kwargs = {'normalize_embeddings': False}
#     embeddings = HuggingFaceBgeEmbeddings(
#         model_name=model_name,
#         model_kwargs=model_kwargs,
#         encode_kwargs=encode_kwargs
#     )


#     global prompt 
#     prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

#     load_vector_store = Chroma(persist_directory="stores/pet_cosine", embedding_function=embeddings)

#     retriever = load_vector_store.as_retriever(search_kwargs={"k":2})

# def RAG():
#     initialize_model()
#     # query = request.form.get('query')
#     # Your logic to handle the query
#     query = "when is the first service"
#     chain_type_kwargs = {"prompt": prompt}
#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         return_source_documents=True,
#         chain_type_kwargs=chain_type_kwargs,
#         verbose=True
#     )
#     response = qa(query)
#     answer = response['result']
#     source_document = response['source_documents'][0].page_content
#     doc = response['source_documents'][0].metadata['source']
#     response_data = {"answer": answer, "source_document": source_document, "doc": doc}
#     print(response)

# if __name__ == "__main__":
#     initialize_store()
#     RAG()
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
import os
import json
from langchain import PromptTemplate, LLMChain
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader

# Define global variables
llm = None
prompt = None
retriever = None

def initialize_store():
    stores_folder = "stores"
    pet_cosine_folder = "pet_cosine"

    stores_path = os.path.join(os.getcwd(), stores_folder)
    pet_cosine_path = os.path.join(stores_path, pet_cosine_folder)

    if os.path.exists(pet_cosine_path) and os.path.isdir(pet_cosine_path):
        print(f"The subfolder '{pet_cosine_folder}' exists inside '{stores_folder}'.")
        print("Proceeding for RAG...")

    else:
        print(f"The subfolder '{pet_cosine_folder}' does not exist inside '{stores_folder}'.")
        print("Proceeding for Vector DB Creation...")
        model_name = "BAAI/bge-large-en"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        loader = DirectoryLoader('data/', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        vector_store = Chroma.from_documents(texts, embeddings, collection_metadata={"hnsw:space": "cosine"}, persist_directory="stores/pet_cosine")

        print("Vector Store Created.......")

def initialize_model():
    global llm
    global prompt
    global retriever

    local_llm = "tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
    config = {
        'max_new_tokens':1024,
        'repetition_penalty':1.1,
        'temperature':0.1,
        'top_k':50,
        'top_p':0.9,
        'stream':True,
        'threads':int(os.cpu_count()/2)
    }

    llm = CTransformers(
        model=local_llm,
        model_type="mistral",
        lib="avx2",
        **config
    )

    print("LLM Initialized....")

    prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """

    model_name = "BAAI/bge-large-en"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

    load_vector_store = Chroma(persist_directory="stores/pet_cosine", embedding_function=embeddings)

    retriever = load_vector_store.as_retriever(search_kwargs={"k":2})

def RAG():
    global llm
    global prompt
    global retriever

    initialize_model()
    
    query = "when is the first service"
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True
    )
    response = qa(query)
    answer = response['result']
    source_document = response['source_documents'][0].page_content
    doc = response['source_documents'][0].metadata['source']
    response_data = {"answer": answer, "source_document": source_document, "doc": doc}
    print(response)

if __name__ == "__main__":
    initialize_store()
    RAG()


    