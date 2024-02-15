import os
import gradio as gr
from langchain import PromptTemplate, LLMChain
from langchain.llms import CTransformers
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.chains import ConversationalRetrievalChain

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
    # Your initialization code for the vector store goes here...

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
initialize_store()
initialize_model()
# Define an empty chat_history list
# chat_history = []

# with gr.Blocks() as demo:
#     chain_type_kwargs = {"prompt": prompt}
#     qa = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=retriever,
#         verbose=True,
#     )
#     chatbot = gr.Chatbot()
#     msg = gr.Textbox()
#     clear = gr.Button("Clear")
    
#     def user(user_message, chat_history):
#         response = qa({"question": user_message, "chat_history": chat_history})
#         chat_history = [(user_message, response["answer"])]
#         return gr.update(value=""), chat_history
    
#     msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)
#     clear.click(lambda: None, None, chatbot, queue=False)



chat_history = []

with gr.Blocks() as demo:
    chain_type_kwargs = {"prompt": prompt}
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        verbose=True,
    )
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    
    def user(user_message, chat_history):
        # Convert chat history to list of tuples
        chat_history_tuples = [(message[0], message[1]) for message in chat_history]
        
        # Get response from QA chain
        response = qa({"question": user_message, "chat_history": chat_history_tuples})
        
        # Append user message and response to chat history
        chat_history.append((user_message, response["answer"]))
        
        return gr.update(value=""), chat_history
    
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)


















# def chatbot(query, history=None):
#     if history is None:
#         history = []

#     # Call the RAG model to generate a response to the input query
#     qa_resp = rag_conversation()  # Assuming rag_conversation is a function that initializes the RAG model
#     response = qa_resp({"question": query, "chat_history": history})

#     # Append the query and its response to the chat history
#     history.append((query, response))

#     # Return the updated chat history
#     return history
# def chatbot(query, history=None):
#     #working
#     if history is None:
#         history = []

#     # Call the RAG model to generate a response to the input query
#     qa_resp = rag_conversation()  # Assuming rag_conversation is a function that initializes the RAG model
#     response = qa_resp({"question": query, "chat_history": history})

#     # Append the query and its response to the chat history
#     history.append((query, response))

#     # Extract the answer/response from the response dictionary
#     answer = response.get("answer", "Sorry, I don't have an answer to that.")

#     # Create the message in the expected format
#     message = [query, answer]

#     # Return the message and the updated chat history as a list of lists
#     return [message], history
# def chatbot(query, history=None):
#     if history is None:
#         history = []

#     # Call the RAG model to generate a response to the input query
#     qa_resp = rag_conversation()  # Assuming rag_conversation is a function that initializes the RAG model
#     response = qa_resp({"question": query, "chat_history": history})

#     # Append the query and its response to the chat history
#     history.append((query, response))

#     # Extract the answer/response from the response dictionary
#     answer = response.get("answer", "Sorry, I don't have an answer to that.")

#     # Create the message in the expected format
#     message = [query, answer]

#     # Return the message and the updated chat history as a list of lists
#     return [message], history

# def chatbot(query, history=None):
#EORKING 2
#     if history is None:
#         history = []

#     # Convert history to list of tuples if it's not already
#     chat_history_tuples = []
#     for message in history:
#         chat_history_tuples.append((message[0], message[1]))

#     # Call the RAG model to generate a response to the input query
#     qa_resp = rag_conversation()  # Assuming rag_conversation is a function that initializes the RAG model
#     response = qa_resp({"question": query, "chat_history": chat_history_tuples})

#     # Append the query and its response to the chat history
#     history.append((query, response))

#     # Extract the answer/response from the response dictionary
#     answer = response.get("answer", "Sorry, I don't have an answer to that.")

#     # Create the message in the expected format
#     message = [query, answer]

#     # Return the message and the updated chat history as a list of lists
#     return [message], history
# def chatbot(query, history=None):
#     if history is None:
#         history = []

#     # Convert history to list of tuples if it's not already
#     chat_history_tuples = []
#     for message in history:
#         chat_history_tuples.append((message[0], message[1]))

#     print("Chat history tuples:", chat_history_tuples)  # Debug print

#     # Call the RAG model to generate a response to the input query
#     qa_resp = rag_conversation()  # Assuming rag_conversation is a function that initializes the RAG model
#     response = qa_resp({"question": query, "chat_history": chat_history_tuples})

#     print("Response:", response)  # Debug print
#     print("Type of response:", type(response))  # Debug print

#     # Append the query and its response to the chat history
#     history.append((query, response))

#     # Extract the answer/response from the response dictionary
#     answer = response.get("answer", "Sorry, I don't have an answer to that.")

#     # Create the message in the expected format
#     message = [query, answer]

#     # Return the message and the updated chat history as a list of lists
#     return [message], history

# def chatbot(query, history=None):
#     if history is None:
#         history = []

#     # Convert history to list of tuples if it's not already
#     chat_history_tuples = []
#     for message in history:
#         chat_history_tuples.append((message[0], message[1]))

#     print("Chat history tuples:", chat_history_tuples)  # Debug print

#     # Call the RAG model to generate a response to the input query
#     qa_resp = rag_conversation()  # Assuming rag_conversation is a function that initializes the RAG model
#     response = qa_resp({"question": query, "chat_history": chat_history_tuples})

#     print("Response:", response)  # Debug print

#     # Extract the answer from the response
#     answer = response["answer"]

#     # Concatenate the query with the answer to form the response message
#     response_message = f"{query}\n{answer}"

#     # Append the query and its response to the chat history
#     history.append((query, response_message))

#     # Return the response message and the updated chat history
#     return response_message, history
# def chatbot(query, history=None):
#     if history is None:
#         history = []

#     # Convert history to list of tuples if it's not already
#     chat_history_tuples = []
#     for message in history:
#         chat_history_tuples.append((message[0], message[1]))

#     # Call the RAG model to generate a response to the input query
#     qa_resp = rag_conversation()  # Assuming rag_conversation is a function that initializes the RAG model
#     response = qa_resp({"question": query, "chat_history": chat_history_tuples})

#     # Extract the answer from the response
#     answer = response["answer"]

#     # Concatenate the query with the answer to form the response message
#     response_message = f"{query}\n{answer}"

#     # Append the query and its response to the chat history
#     updated_history = history + [(query, answer)]

#     # Return the response message and the updated chat history as a list of lists
#     return [[query, response_message]], updated_history

# def postprocess(chatbot_output):
#     # Extract the response message from the chatbot output
#     response_message = chatbot_output[0][0][1]
#     return response_message


if __name__ == "__main__":
    demo.launch(debug=True)
    # Launch Gradio interface
    # gr.Interface(fn=chatbot,
    #              inputs=["text", "state"],
    #              outputs=["chatbot", "state"]).launch()
