from haystack.components.embedders import OpenAITextEmbedder
from haystack.utils import Secret
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from colorama import Fore, Style, init

# Initialize Colorama
init(autoreset=True)

# Define the index name
index_name = "SIBS"

# Initialize the document embedder
document_embedder = OpenAITextEmbedder(api_key=Secret.from_env_var("OPENAI_API_KEY"))

# Initialize the Qdrant document store
document_store = QdrantDocumentStore(
    url="localhost",
    index=index_name,
    embedding_dim=1536,
    return_embedding=True,
    wait_result_from_api=True,
)

# Initialize the retriever
retriever = QdrantEmbeddingRetriever(
    document_store=document_store,
    top_k=5,
    scale_score=True,
    return_embedding=True,
)

# Initialize the OpenAI chat generator
chat_generator = OpenAIChatGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"), model="gpt-3.5-turbo")

while True:
    
    print(f"{Fore.WHITE}{'-'*80}")

    # Get the user query
    query = input(f"{Fore.GREEN}Enter your query or type 'exit' to quit: {Style.RESET_ALL}")
    if query.lower() == 'exit':
        break

    # Embed the query
    query_embedding = document_embedder.run(query)["embedding"]

    # Retrieve documents based on the query embedding
    retrieval_result = retriever.run(query_embedding=query_embedding)
    documents = retrieval_result["documents"]

    # Format documents for LLM input
    doc_contents = "\n".join([doc.content for doc in documents]) if documents else "No relevant documents found."

    # Run the chat generator
    prompt = f"Based on these documents:\n\n{doc_contents}\n\nAnswer the user query:\n{query}"
    result = chat_generator.run([ChatMessage.from_user(prompt)])

    # Print the generated answer
    print(f"\n{Fore.YELLOW}Response:{Style.RESET_ALL} {result['replies'][0].content}")
