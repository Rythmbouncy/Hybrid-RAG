import os
from dotenv import load_dotenv
from src.dataloader import load_and_prepare_data
from src.encoders import initialize_encoders
from src.pinecone import initialize_pinecone, upsert_data
from src.rag import hybrid_query, generate_rag_response

def main():
    load_dotenv()

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    INDEX_NAME = os.getenv("INDEX_NAME")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION"))
    PINECONE_CLOUD = os.getenv("PINECONE_CLOUD")
    PINECONE_REGION = os.getenv("PINECONE_REGION")
    
    if not all([PINECONE_API_KEY, GEMINI_API_KEY, INDEX_NAME]):
        print("Error: Missing API keys or configuration in .env file.")
        return

    texts, ids = load_and_prepare_data()
    dense_encoder, bm25_encoder = initialize_encoders(EMBEDDING_MODEL, texts)
    index = initialize_pinecone(PINECONE_API_KEY, INDEX_NAME, EMBEDDING_DIMENSION, PINECONE_CLOUD, PINECONE_REGION)
    
    if index.describe_index_stats().get('total_vector_count', 0) == 0:
        upsert_data(index, ids, texts, dense_encoder, bm25_encoder)
    else:
        print("Index already populated. Skipping upsert.")
        
    #Enter query for rag->    
    test_query = "can u tell me what are the things that affect biology systems?"
    top_k_chunks = 3
    alpha_weight = 0.6 
    
    print("\nStarting Hybrid Retrieval->")
    retrieved_contexts, retrieval_latency = hybrid_query(
        test_query, index, dense_encoder, bm25_encoder, top_k=top_k_chunks, alpha=alpha_weight
    )
    for i, context in enumerate(retrieved_contexts):
        print(f"\nDOCUMENT {i+1}->")
        print(context)
    print(f"Retrieval Latency: {retrieval_latency:.4f} seconds")
        
    print("\nGemini RAG Generation->")
    final_response, llm_latency = generate_rag_response(test_query, retrieved_contexts, GEMINI_API_KEY)
    total_latency = retrieval_latency + llm_latency
    
    print(f"LLM Generation Latency: {llm_latency:.4f} seconds")
    print(f"Total Average Query Latency: {total_latency:.4f} seconds")
    print("\nFinal Answer from Gemini->")
    print(final_response)

if __name__ == "__main__":
    main()