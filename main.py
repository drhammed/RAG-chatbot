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


# Setup - Streamlit secrets
#since I am deploying this locally, I don't need the st_secret. So, bypass this

OPENAI_API_KEY = st.secrets["api_keys"]["OPENAI_API_KEY"]
VOYAGE_AI_API_KEY = st.secrets["api_keys"]["VOYAGE_AI_API_KEY"]
PINECONE_API_KEY = st.secrets["api_keys"]["PINECONE_API_KEY"]
aws_access_key_id = st.secrets["aws"]["aws_access_key_id"]
aws_secret_access_key = st.secrets["aws"]["aws_secret_access_key"]
aws_region = st.secrets["aws"]["aws_region"]

#load_dotenv()
# # Initialize Pinecone
# PINECONE_API_KEY = os.getenv('My_Pinecone_API_key')
# # Initialize OpenAI
# OPENAI_API_KEY = os.getenv('My_OpenAI_API_key')
# # Initialize VoyageAI
# VOYAGE_AI_API_KEY = os.getenv("My_voyageai_API_key")
# #Initialize the GroqAPI
# GROQ_API_KEY = os.getenv("My_Groq_API_key")
# #aws
# aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
# aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
# aws_region = os.getenv('AWS_REGION')


# Langchain stuff
# OpenAI model
#llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o",temperature=0.02,max_tokens=None,timeout=None,max_retries=2,api_key=OPENAI_API_KEY)


#Groq model
#model = 'llama3-70b-8192'
#model = 'gemma2-9b-it'
# Initialize Groq Langchain chat object and conversation
#llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model, temperature=0.02)



# Ignore all warnings
warnings.filterwarnings("ignore")

# Set up Streamlit app
st.set_page_config(page_title="Custom Chatbot", layout="wide")
st.title("Custom Chatbot with Retrieval Abilities")

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
    prompt = f"Instruction: You are a helpful assistant to help users with their patient education queries. \
               Based on the following information, provide a summarized & concise explanation using a couple of sentences. \
               Only respond with the information relevant to the user query {query}, \
               if there are none, make sure you say the `magic words`: 'I don't know, I did not find the relevant data in the knowledge base.' \
               But you could carry out some conversations with the user to make them feel welcomed and comfortable, in that case you don't have to say the `magic words`. \
               In the event that there's relevant info, make sure to attach the download button at the very end: \n\n[More Info]({s3_gen_url}) \
               Context: {combined_content}"
    
    # Originally there were no message
    message = HumanMessage(content=prompt)

    response = llm([message])
    return response

# Function to save chat history to a file
def save_chat_history_to_file(filename, history):
    with open(filename, 'w') as file:
        file.write(history)

# Function to upload the file to S3
def upload_file_to_s3(bucket, key, filename):
    s3_client.upload_file(filename, bucket, key)

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


    



# Initialize the conversation memory
memory = ConversationBufferMemory()

prompt_template = ChatPromptTemplate.from_template(
        "Instruction: You are a helpful assistant to help users with their patient education queries. \
        Based on the following information, provide a summarized & concise explanation using a couple of sentences. \
        Only respond with the information relevant to the user query {query}, \
        if there are none, make sure you say the `magic words`: 'I don't know, I did not find the relevant data in the knowledge base.' \
        But you could carry out some conversations with the user to make them feel welcomed and comfortable, in that case you don't have to say the `magic words`. \
        In the event that there's relevant info, make sure to attach the download button at the very end: \n\n[More Info]({s3_gen_url}) \
        Context: {combined_content}"
    )

# Initialize necessary objects (s3 client, Pinecone, OpenAI, etc.)
s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

# Initialize Pinecone
#pc = Pinecone(api_key=os.getenv("My_Pinecone_API_key"))
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)
## HARD CODED index names and host
index_name = "diabetes-ind"
#index = pc.Index(index_name, host="https://diabetes-ind-3w8l5y1.svc.aped-4627-b74a.pinecone.io")

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# Set up LangChain objects
# VOYAGE AI
model_name = "voyage-large-2"  
embedding_function = VoyageAIEmbeddings(
    model=model_name,  
    voyage_api_key=VOYAGE_AI_API_KEY
)
# Initialize the Pinecone client
vector_store = PineconeVectorStore.from_existing_index(
    embedding=embedding_function,
    index_name=index_name
)
retriever = vector_store.as_retriever()

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