A comprehensive chatbot application that integrates advanced retrieval capabilities to provide contextual and relevant information on Diabetes. The chatbot leverages various APIs to enhance its functionality and ensure accurate and efficient responses.

The application uses the Groq API to generate responses and leverages LangChain's [ConversationBufferWindowMemory](https://python.langchain.com/v0.1/docs/modules/memory/types/buffer_window/) to maintain a history of the conversation, providing context for the chatbot's responses.

**Features**

Conversational Interface: The application provides a conversational interface where users can ask questions or make statements, and the chatbot responds accordingly.

Contextual Responses: The application maintains a history of the conversation, which is used to provide relevant and context for the chatbot's responses.

LangChain Integration: The chatbot is powered by the LangChain API, which uses advanced natural language processing techniques to generate human-like responses.

Retrieval Augmented Generation (RAG): The chatbot retrieves relevant documents from a Pinecone vector store and uses them to generate accurate and contextually relevant responses.

Advanced Retrieval: Utilizes a powerful retrieval system to fetch relevant documents and information.

Multi-API Integration: Integrates various APIs including Groq, VoyageAI, Pinecone, and OpenAI to enhance chatbot capabilities.

Presigned URLs: The application generates presigned URLs for S3-stored documents, allowing users to access additional resources easily.



**Usage**

Initialize Objects: Set up necessary objects including S3 client, OpenAI API, and vector store.

Conversational Memory: Use ConversationBufferWindowMemory to maintain a conversation history.

Retrieve and Format Responses: Use the retriever and Groq model to get relevant documents and generate responses.

Save and Upload Chat History: Save chat history to a file and upload it to S3, generating a presigned URL for download.

