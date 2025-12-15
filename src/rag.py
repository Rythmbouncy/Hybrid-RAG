import time
from google import genai
from pinecone import Pinecone

def hybrid_query(query: str, index: Pinecone.Index, dense_encoder, bm25_encoder, top_k: int = 5, alpha: float = 0.5):
    start_time = time.time()
    

    dense_vector = dense_encoder.encode(query).tolist()
    sparse_vector = bm25_encoder.encode_queries(query)

    results = index.query(
        vector=dense_vector,
        sparse_vector=sparse_vector,
        top_k=top_k,
        hybrid_search_config={"alpha": alpha},
        include_metadata=True
    )

    retrieval_latency = time.time() - start_time
    
    contexts = [match['metadata']['text'] for match in results['matches']]
    
    return contexts, retrieval_latency

def generate_rag_response(query: str, contexts: list, gemini_api_key: str):
    
    try:
        gemini_client = genai.Client(api_key=gemini_api_key)
    except Exception as e:
        return f"Error initializing Gemini client: {e}", 0

    instruction = (
        "You are an expert academic assistant. Use ONLY the provided context to answer the user's question. "
        "If the answer is not found in the context, state 'The provided documents do not contain enough information to answer this question.'"
    )
    
    context_text = "\n\n---\n\n".join(f"DOCUMENT {i+1}:\n{c}" for i, c in enumerate(contexts))
    
    prompt = f"Context:\n{context_text}\n\nQuestion: {query}"
    
    llm_start_time = time.time()
    
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={"system_instruction": instruction}
        )
        llm_latency = time.time() - llm_start_time
        return response.text, llm_latency
    except Exception as e:
        return f"LLM Generation Error: {e}", 0