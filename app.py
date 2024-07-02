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
from pinecone import Pinecone
import pinecone
from langchain_openai import ChatOpenAI
import openai
from groq import Groq
from langchain.chains import LLMChain, RetrievalQA
import time
import re
import warnings
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

# Load environment variables from Streamlit secrets
OPENAI_API_KEY = st.secrets["api_keys"]["OPENAI_API_KEY"]
VOYAGE_AI_API_KEY = st.secrets["api_keys"]["VOYAGE_AI_API_KEY"]
PINECONE_API_KEY = st.secrets["api_keys"]["PINECONE_API_KEY"]
GROQ_API_KEY = st.secrets["api_keys"]["GROQ_API_KEY"]
aws_access_key_id = st.secrets["aws"]["aws_access_key_id"]
aws_secret_access_key = st.secrets["aws"]["aws_secret_access_key"]
aws_region = st.secrets["aws"]["aws_region"]

# Initialize necessary objects
s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

openai.api_key = OPENAI_API_KEY

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
model = 'llama3-70b-8192'
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model, temperature=0.02)

# Simplified prompt template
prompt_template = ChatPromptTemplate.from_template(
    "You are an assistant providing concise and relevant information for patient education queries. Only respond with relevant information or indicate if none is found."
)

# Initialize the conversation memory
memory = ConversationBufferMemory()

# Setup retry with backoff function
def retry_with_backoff(api_call, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            return api_call()
        except openai.error.RateLimitError as e:
            wait_time = (2 ** retries) * 60  # Exponential backoff
            print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1
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
def retrieve_and_format_response(query, retriever, llm):
    docs = retriever.get_relevant_documents(query)
    
    formatted_docs = []
    for doc in docs:
        content_data = doc.page_content
        s3_uri = doc.metadata['id']
        s3_gen_url = generate_presigned_url(s3_uri)
        formatted_doc = f"{content_data}\n\n[More Info]({s3_gen_url})"
        formatted_docs.append(formatted_doc)
    
    combined_content = "\n\n".join(formatted_docs)
    
    # Create a prompt for the LLM to generate an explanation based on the retrieved content
    prompt = f"Instruction: Based on the following information, provide a summarized & concise explanation using a couple of sentences relevant to the query '{query}'.\n\nContext: {combined_content}"
    
    message = HumanMessage(content=prompt)

    response = retry_with_backoff(lambda: llm([message]))
    return response

# Initialize rag_chain
rag_chain = (
    {"retrieved_context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

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
