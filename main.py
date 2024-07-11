#pip install langchain_voyageai
#!pip install langchain_openai
#!pip install langchain_pinecone
#pip install groq
#!pip install langchain_groq


import streamlit as st
from langchain_voyageai import VoyageAIEmbeddings
import os
import json
import boto3
from dotenv import load_dotenv
from urllib.parse import urlparse
import pinecone
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
import openai
from groq import Groq
from langchain.chains import LLMChain, RetrievalQA
import time
import re
import warnings
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import Runnable
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import uuid
from openai import OpenAIError



# Set up Streamlit app
st.set_page_config(page_title="Custom Chatbot", layout="wide")
st.title("Custom Chatbot with Retrieval Abilities")


# Load environment variables from Streamlit secrets
OPENAI_API_KEY = st.secrets["api_keys"]["OPENAI_API_KEY"]
VOYAGE_AI_API_KEY = st.secrets["api_keys"]["VOYAGE_AI_API_KEY"]
PINECONE_API_KEY = st.secrets["api_keys"]["PINECONE_API_KEY"]
GROQ_API_KEY = st.secrets["api_keys"]["GROQ_API_KEY"]
aws_access_key_id = st.secrets["aws"]["aws_access_key_id"]
aws_secret_access_key = st.secrets["aws"]["aws_secret_access_key"]
aws_region = st.secrets["aws"]["aws_region"]

# Initialize Pinecone
Pinecone(api_key=PINECONE_API_KEY,host="https://diabetes-ind-3w8l5y1.svc.aped-4627-b74a.pinecone.io")
index_name = "diabetes-ind"
#index = pc.Index(index_name, host="https://diabetes-ind-3w8l5y1.svc.aped-4627-b74a.pinecone.io")


openai.api_key = OPENAI_API_KEY


# Initialize necessary objects
s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)


model_name = "voyage-large-2"
embedding_function = VoyageAIEmbeddings(
    model=model_name,  
    voyage_api_key=VOYAGE_AI_API_KEY
)


# Initialize Pinecone
vector_store = PineconeVectorStore.from_existing_index(
    embedding=embedding_function,
    index_name="diabetes-ind"
)


retriever = vector_store.as_retriever()

# Groq model
#model = 'llama3-70b-8192'
#model = 'llama3-8b-8192'
#llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model, temperature=0.02)

#OpenAI model
llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)

# Simplified prompt template
prompt_template = ChatPromptTemplate.from_template(
    "You are an assistant providing concise and relevant information for patient education queries. Only respond with relevant information or indicate if none is found."
)

# Set up conversation buffer memory with a window
conversational_memory_length = 5
memory = ConversationBufferWindowMemory(
    k=conversational_memory_length,
    memory_key="chat_history",
    return_messages=True
)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Setup retry with backoff function
def retry_with_backoff(api_call, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            return api_call()
        except OpenAIError as e:
            if isinstance(e, openai.error.RateLimitError):
                wait_time = (2 ** retries) * 60  # Exponential backoff
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
            else:
                raise e
    raise Exception("Maximum retries exceeded")

# Function to generate pre-signed URL
def generate_presigned_url(s3_uri):
    parsed_url = urlparse(s3_uri)
    bucket_name = parsed_url.netloc
    object_key = parsed_url.path.lstrip('/')
    presigned_url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket_name, 'Key': object_key},
        ExpiresIn=3600
    )
    return presigned_url

# Function to retrieve documents, generate URLs, and format the response

def retrieve_and_format_response(query, retriever, llm, max_docs=5, max_chars=1000):
    docs = retriever.get_relevant_documents(query)[:max_docs]
    
    formatted_docs = []
    total_length = 0
    for doc in docs:
        content_data = doc.page_content
        s3_uri = doc.metadata['id']
        s3_gen_url = generate_presigned_url(s3_uri)
        formatted_doc = f"{content_data}\n\n[More Info]({s3_gen_url})"
        
        if total_length + len(formatted_doc) > max_chars:
            break
        
        formatted_docs.append(formatted_doc)
        total_length += len(formatted_doc)
    
    combined_content = "\n\n".join(formatted_docs)
    
    # Create a prompt for the LLM to generate an explanation based on the retrieved content
    prompt = f"Instruction: Based on the following information, provide a summarized & concise explanation using a couple of sentences relevant to the query '{query}'.\n\nContext: {combined_content}"
    
    message = HumanMessage(content=prompt)

    response = retry_with_backoff(lambda: llm([message]))
    return response

# Function to save chat history to a file

def save_chat_history_to_file(filename, history):
    with open(filename, 'w') as file:
        json.dump(history, file)
        
# Function to upload the file to S3
def upload_file_to_s3(bucket, key, filename):
    s3_client.upload_file(filename, bucket, key)
    return generate_presigned_url(f"s3://{bucket}/{key}")

# Example usage with memory
def ask_question(query, chain, llm):
    # Retrieve and format the response with pre-signed URLs
    response_with_docs = retrieve_and_format_response(query, retriever, llm)
    
    # Add the retrieved response to the memory
    memory.save_context({"input": query}, {"output": response_with_docs['answer']})
    
    # Use the conversation chain to get the final response
    response = chain.invoke(query)
    pattern = r"s3(.*?)(?=json)"
    s3_uris = ["s3" + x + "json" for x in re.findall(pattern, response)]
    for s3_uri in s3_uris:
        final_response = response.replace(s3_uri, generate_presigned_url(s3_uri))
    return final_response


# Initialize rag_chain
rag_chain = (
    {"retrieved_context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Define the S3 bucket and file details
bucket_name = "chatbot-pro"
folder_name = "chat-history"
file_key = f"{folder_name}/chat_history_{uuid.uuid4()}.json"


# Display chat messages from history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
user_input = st.chat_input("You: ")

if user_input:
    # Add user message to chat history
    st.session_state["messages"].append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate and display bot response
    with st.spinner("Thinking..."):
        bot_response = retrieve_and_format_response(user_input, retriever, llm).content
    
    st.session_state["messages"].append({"role": "assistant", "content": bot_response})
    
    with st.chat_message("assistant"):
        st.markdown(bot_response)
        
    # Save chat history to a file
    filename = "chat_history.json"
    save_chat_history_to_file(filename, st.session_state["messages"])
    
    # Upload the file to S3 and get the pre-signed URL
    presigned_url = upload_file_to_s3(bucket_name, file_key, filename)
    
    # Display download link for chat history
    #st.markdown(f"[Download Chat History]({presigned_url})")
