from flask import Flask, request, jsonify, render_template
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
from langchain.chains import LLMChain, RetrievalQA
import time
import re
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

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load environment variables from .env file
load_dotenv()
# Initialize Pinecone
PINECONE_API_KEY = os.getenv('My_Pinecone_API_key')
# Initialize OpenAI
OPENAI_API_KEY = os.getenv('My_OpenAI_API_key')
# Initialize VoyageAI
VOYAGE_AI_API_KEY = os.getenv("My_voyageai_API_key")
# Initialize the GroqAPI
GROQ_API_KEY = os.getenv("My_Groq_API_key")
# AWS key
aws_access_key_id = os.getenv('aws_access_key_id')
aws_secret_access_key = os.getenv('aws_secret_access_key')
aws_region = os.getenv('aws_region')

# Initialize necessary objects
s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

# Initialize Pinecone
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)
# HARD CODED index names and host
index_name = "diabetes-ind"
# index = pc.Index(index_name, host="https://diabetes-ind-3w8l5y1.svc.aped-4627-b74a.pinecone.io")

openai.api_key = OPENAI_API_KEY

model_name = "voyage-large-2"
embedding_function = VoyageAIEmbeddings(
    model=model_name,
    voyage_api_key=VOYAGE_AI_API_KEY
)

# Initialize Pinecone vector store
vector_store = PineconeVectorStore.from_existing_index(
    embedding=embedding_function,
    index_name="diabetes-ind"
)

retriever = vector_store.as_retriever()

# OpenAI model
#llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)

# OpenAI model
#llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o",temperature=0,max_tokens=None,timeout=None,max_retries=2,api_key=OPENAI_API_KEY)
#Groq model
#model = 'llama3-70b-8192'
#model = 'gemma2-9b-it'
#model = 'llama3-8b-8192'
# Initialize Groq Langchain chat object and conversation
#llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model, temperature=0.02)

# Simplified prompt template
# prompt_template = ChatPromptTemplate.from_template(
#     """
#     You are an assistant providing concise and relevant information for patient education queries about diabetes.
#     If the user greets you, responds with a friendly greeting.
#     If the user introduces themselves, respond politely and warmly.
#     If the user expresses gratitude, respond graciously.
#     If the user gives a compliment, respond appreciatively.
#     If the user expresses humor or laughter, acknowledge it kindly.
#     If the user asks a general question, respond appropriately.
#     If the user asks a question specifically related to diabetes, provide a concise and relevant answer.
#     If no relevant information is found, politely indicate that.
#     """
# )

prompt_template = ChatPromptTemplate.from_template(
        "Instruction: You are a helpful assistant to help users with their patient education queries. \
        Based on the following information, provide a summarized & concise explanation using a couple of sentences. \
        Only respond with the information relevant to the user query {query}, \
        if there are none, make sure you say the `magic words`: 'I don't know, I did not find the relevant data in the knowledge base.' \
        But you could carry out some conversations with the user to make them feel welcomed and comfortable, in that case you don't have to say the `magic words`. \
        In the event that there's relevant info, make sure to attach the download button at the very end: \n\n[More Info]({s3_gen_url}) \
        Context: {combined_content}"
    ) 


# Set up conversation buffer memory with a window
conversational_memory_length = 5
memory = ConversationBufferWindowMemory(
    k=conversational_memory_length,
    memory_key="chat_history",
    return_messages=True
)

# Initialize chat history
chat_history = []

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
    #prompt = f"Instruction: Based on the following information, provide a summarized & concise explanation using a couple of sentences relevant to the query '{query}'.\n\nContext: {combined_content}"
    
    prompt = f"""Instruction: You are a helpful assistant to help users with their patient education queries. 
Based on the following information, provide a summarized & concise explanation using a couple of sentences. 
Only respond with the information relevant to the user query {query}, 
if there are none, make sure you say the `magic words`: 'I don't know, I did not find the relevant data in the knowledge base.' 
But you could carry out some conversations with the user to make them feel welcomed and comfortable, in that case you don't have to say the `magic words`. 
In the event that there's relevant info, make sure to attach the download button at the very end: \n\n[More Info]({s3_gen_url}) 
Context: {combined_content}"""


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

# Define the S3 bucket and file details
bucket_name = "chatbot-pro"
folder_name = "chat-history"
file_key = f"{folder_name}/chat_history_{uuid.uuid4()}.json"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get("message").strip().lower()

    compliments = ["good job", "good job" , "fantastic", "awesome", "great", "well done", "nice work", "impressed", "amazing", "superb", "legit"]
    greetings = ["hi", "hello", "hey"]
    gratitude = ["thank", "thanks", "thank you"]
    humor = ["lol", "haha", "hehe", "lmao"]
    affection = ["i like you", "i love you", "you're great", "you're amazing", "you're fantastic"]

    if user_input:
        # Add user message to chat history
        chat_history.append({"role": "user", "content": user_input})

        # Generate and display bot response
        if any(greet in user_input for greet in greetings):
            name_match = re.search(r"(my name is|i am|i'm)\s+(\w+)", user_input)
            if name_match:
                name = name_match.group(2).capitalize()
                bot_response = f"Hello {name}! How can I assist you today? Feel free to ask me about diabetes or any related information."
            else:
                bot_response = "Hello! How can I assist you today? Feel free to ask me about diabetes or any related information."
        elif any(phrase in user_input for phrase in affection):
            bot_response = "Thank you for your kind words! How can I assist you with diabetes-related information?"
        elif any(thank in user_input for thank in gratitude):
            bot_response = "You're welcome! If you have any other questions about diabetes or need further assistance, feel free to ask."
        elif any(comp in user_input for comp in compliments):
            bot_response = "Thank you! I'm here to help. How can I assist you further with diabetes-related information?"
        elif any(hum in user_input for hum in humor):
            bot_response = "Glad you found it funny! How can I assist you with diabetes-related information?"
        elif any(intro in user_input for intro in ["my name is", "i am", "i'm"]) and not any(comp in user_input for comp in compliments):
            name_match = re.search(r"(my name is|i am|i'm)\s+(\w+)", user_input)
            if name_match:
                name = name_match.group(2).capitalize()
                bot_response = f"Nice to meet you, {name}! How can I assist you today with information about diabetes or related topics?"
            else:
                bot_response = "Nice to meet you! How can I assist you today with information about diabetes or related topics?"
        else:
            bot_response = retrieve_and_format_response(user_input, retriever, llm).content

        chat_history.append({"role": "assistant", "content": bot_response})

        # Save chat history to a file
        filename = "chat_history.json"
        save_chat_history_to_file(filename, chat_history)

        # Upload the file to S3 and get the pre-signed URL
        presigned_url = upload_file_to_s3(bucket_name, file_key, filename)

        return jsonify({
            "response": bot_response,
            "chat_history_url": presigned_url
        })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
