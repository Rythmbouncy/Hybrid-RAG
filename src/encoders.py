from pinecone_text.sparse import BM25Encoder
from sentence_transformers import SentenceTransformer

def initialize_encoders(model_name: str, docs: list):

    dense_encoder = SentenceTransformer(model_name)
    
    bm25_encoder = BM25Encoder()
    bm25_encoder.fit(docs)
    
    return dense_encoder, bm25_encoder

if __name__ == '__main__':
    DUMMY_DOC = ["The cat sat on the mat.", "The dog ran after the ball.", "A cat chased a dog."]
    dense, sparse = initialize_encoders('all-MiniLM-L6-v2', DUMMY_DOC)