import time
from pinecone import Pinecone,ServerlessSpec
from tqdm.auto import tqdm
from pinecone import Vector

def initialize_pinecone(api_key: str, index_name: str, dimension: int, cloud: str, region: str):
    pc = Pinecone(api_key=api_key)
    
    if index_name not in pc.list_indexes().names():
        print(f"Creating Pinecone index: {index_name}...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="dotproduct",
            spec=ServerlessSpec(cloud='aws',region="us-east-1")
        )
        print("Index created.")
    else:
        print(f"Connecting to existing index: {index_name}")

    return pc.Index(index_name)

def upsert_data(index, ids: list, texts: list, dense_encoder, bm25_encoder, batch_size: int = 64):
    
    print(f"\nStarting hybrid upsert for {len(texts)} documents in batches of {batch_size}...")

    for i in tqdm(range(0, len(texts), batch_size)):
        i_end = min(i + batch_size, len(texts))
        
        batch_texts = texts[i:i_end]
        batch_ids = ids[i:i_end]
        
        dense_vectors = dense_encoder.encode(batch_texts).tolist()
        sparse_dicts = bm25_encoder.encode_documents(batch_texts) 

        batch_vectors = []
        for doc_id, text, dense_vec, sparse_dict in zip(batch_ids, batch_texts, dense_vectors, sparse_dicts):
            vector_obj = Vector(
                id=doc_id,
                values=dense_vec,
                metadata={"text": text},
                sparse_values={
                    "indices": sparse_dict['indices'],
                    "values": sparse_dict['values']
                }
            )
            batch_vectors.append(vector_obj)
        
        index.upsert(vectors=batch_vectors)

    time.sleep(5) 
    final_count = index.describe_index_stats().get('total_vector_count')
    print(f"\n Upsert completed. Final vector count in index: {final_count}")

if __name__ == '__main__':
    print("Pinecone check.")