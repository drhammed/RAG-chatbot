A RAG-based Diabetes information chatbot that leverages advanced retrieval capabilities to provide contextual and accurate patient education on Diabetes. The application supports multiple LLM providers and conversation modes to deliver flexible and efficient responses.

**Features**

**Dual Conversation Modes:**
- **Single Question Mode**: Quick Q&A without conversation history for independent queries
- **Chat Mode**: Context-aware conversations with memory of the last 10 messages (5 exchanges)

**Multiple LLM Provider Support:**
- **Ollama (Cloud)**: Access to powerful cloud-hosted models including:
  - gpt-oss:20b
  - gpt-oss:120b
  
- **Groq**: Fast inference with models like:
  - meta-llama/llama-4-scout-17b-16e-instruct
  - groq/compound-mini

**Retrieval Augmented Generation (RAG)**: Retrieves relevant diabetes education documents from a Pinecone vector store and generates accurate, contextually relevant responses based on the knowledge base.

**VoyageAI Embeddings**: Uses voyage-large-2 model for high-quality semantic search and document retrieval.

**AWS S3 Integration**: Generates presigned URLs for documents stored in S3, allowing users to access additional resources and download full materials easily.

**Streamlit Web Interface**: User-friendly interface with model selection, conversation mode switching, and real-time chat functionality.



**Usage**

1. **Initialize Objects**: Set up necessary components including S3 client, Pinecone vector store, and VoyageAI embeddings.

2. **Select Model Provider**: Choose between Ollama (Cloud) or Groq models from the sidebar.

3. **Choose Conversation Mode**: Select "Single Question" for independent queries or "Chat" for context-aware conversations.

4. **Query the Chatbot**: Ask questions about diabetes, and the chatbot will retrieve relevant information from the knowledge base and generate concise, informative responses.

5. **Access Additional Resources**: Click on the "More Info" links in responses to download full documents from S3 via presigned URLs.

