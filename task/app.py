from task._constants import API_KEY
from task.chat.chat_completion_client import DialChatCompletionClient
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role


#TODO:
# Create system prompt with info that it is RAG powered assistant.
# Explain user message structure (firstly will be provided RAG context and the user question).
# Provide instructions that LLM should use RAG Context when answer on User Question, will restrict LLM to answer
# questions that are not related microwave usage, not related to context or out of history scope
SYSTEM_PROMPT = """You are a RAG-powered assistant specialized in providing information about microwave usage from the provided documentation.

Your responses are based on Retrieval-Augmented Generation (RAG). Each user query will be accompanied by relevant context from the microwave manual.

Message Structure:
1. RAG Context: Relevant excerpts from the microwave documentation retrieved based on the user's question
2. User Question: The user's actual question

Instructions:
- Use the provided RAG Context to answer the user's question accurately
- Only provide information that is explicitly mentioned in the RAG Context or related to microwave usage
- Do not answer questions that are unrelated to microwave usage or functionality
- Do not provide information that is not covered in the context or outside the scope of microwave documentation
- If the context does not contain relevant information to answer the question, politely inform the user that the information is not available in the documentation
- Be concise and helpful in your responses"""

#TODO:
# Provide structured system prompt, with RAG Context and User Question sections.
USER_PROMPT = """RAG Context:
{rag_context}

User Question:
{user_question}"""


# Initialize clients and services
embeddings_client = DialEmbeddingsClient(
    deployment_name='text-embedding-3-small-1',
    api_key=API_KEY
)

chat_completion_client = DialChatCompletionClient(
    deployment_name='gpt-4o-mini-2024-07-18',
    api_key=API_KEY
)

db_config = {
    'host': 'localhost',
    'port': 5433,
    'database': 'vectordb',
    'user': 'postgres',
    'password': 'postgres'
}

text_processor = TextProcessor(
    embeddings_client=embeddings_client,
    db_config=db_config
)


def run_chat():
    """Run console chat with RAG augmentation"""
    # Ingest the microwave manual into the database
    print("Loading microwave manual into vector database...")
    text_processor.process_text_file(
        file_name='task/embeddings/microwave_manual.txt',
        chunk_size=500,
        overlap=50,
        dimensions=1536,
        truncate=True
    )
    print("Microwave manual loaded successfully!\n")
    
    conversation = Conversation()
    
    # Add system prompt to conversation
    system_message = Message(role=Role.SYSTEM, content=SYSTEM_PROMPT)
    conversation.add_message(system_message)
    
    print("RAG-Powered Microwave Assistant")
    print("================================")
    print("Type your questions about microwave usage. Type 'exit' to quit.\n")
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Retrieve context from vector database
        search_results = text_processor.search(
            search_mode=SearchMode.COSINE_DISTANCE,
            user_request=user_input,
            top_k=3,
            min_score_threshold=0.5,
            dimensions=1536
        )
        # Create RAG context from search results
        rag_context = "\n".join([result['text'] for result in search_results])
        
        # Augment user message with RAG context
        augmented_user_message = USER_PROMPT.format(
            rag_context=rag_context,
            user_question=user_input
        )
        
        # Add user message to conversation
        user_message = Message(role=Role.USER, content=augmented_user_message)
        conversation.add_message(user_message)
        
        # Generate response
        response_content = chat_completion_client.get_completion(conversation.messages)
        
        # Add assistant response to conversation
        assistant_message = Message(role=Role.AI, content=response_content.content)
        conversation.add_message(assistant_message)
        
        print(f"Assistant: {response_content.content}\n")


if __name__ == '__main__':
    run_chat()