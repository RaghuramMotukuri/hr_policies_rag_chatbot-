import os
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from typing import List, Any
import numpy as np

class FirestoreVectorStore:
    def __init__(self, collection_name: str = "hr_policies_embeddings"):
        self.collection_name = collection_name
        self._init_firebase()
        self.db = firestore.client()
        self.collection = self.db.collection(self.collection_name)
        self.model = None
        self.local_cache = None
        self.native_search_failed = False
        print(f"[INFO] Initialized FirestoreVectorStore for collection: {self.collection_name}")

    def _init_firebase(self):
        if not firebase_admin._apps:
            # Load env variables (if loaded)
            from dotenv import load_dotenv
            load_dotenv()
            cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json")
            print(f"[INFO] Initializing Firebase with credential: {cred_path}")
            if not os.path.exists(cred_path):
                raise FileNotFoundError(f"Firebase credentials file not found at: {cred_path}")
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)

    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        from src.embedding import EmbeddingPipeline
        emb_pipe = EmbeddingPipeline(chunk_size=1000, chunk_overlap=200)
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)

        # Delete all existing documents in the collection to refresh the whole database
        print(f"[INFO] Deleting existing documents in collection '{self.collection_name}' to refresh...")
        try:
            docs = self.collection.get()
            delete_batch = self.db.batch()
            for k, doc in enumerate(docs):
                delete_batch.delete(doc.reference)
                if (k + 1) % 500 == 0:
                    delete_batch.commit()
                    delete_batch = self.db.batch()
            delete_batch.commit()
            print(f"[INFO] Successfully deleted {len(docs)} existing documents.")
        except Exception as e:
            print(f"[WARNING] Failed to clear collection: {e}")

        print(f"[INFO] Uploading {len(chunks)} chunks to Firestore...")
        # Write to Firestore in a batch to be efficient
        batch = self.db.batch()
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc_id = f"chunk_{i:04d}"
            doc_ref = self.collection.document(doc_id)
            vector_data = Vector(embedding.tolist())
            
            # Enrich metadata with the page content
            metadata = dict(chunk.metadata)
            
            doc_data = {
                "text": chunk.page_content,
                "embedding": vector_data,
                "metadata": metadata
            }
            batch.set(doc_ref, doc_data)
            
            # Commit batches in chunks of 500 (Firestore batch limit)
            if (i + 1) % 500 == 0:
                batch.commit()
                batch = self.db.batch()
                print(f"[INFO] Committed {i + 1} chunks...")
        
        batch.commit()
        self.local_cache = None
        print(f"[INFO] Vector store built and saved to Firestore collection: {self.collection_name}")

    def save(self):
        # No-op for Firestore (data is persistent on upload)
        pass

    def load(self):
        # No-op for Firestore (data is persistent on upload)
        pass

    def search(self, query_embedding: List[float], top_k: int = 5):
        query_vector = Vector(query_embedding)
        
        # If we have already detected that native search fails, go straight to local cache
        if getattr(self, "native_search_failed", False):
            return self._local_search(query_embedding, top_k)
            
        try:
            results = (
                self.collection.find_nearest(
                    vector_field="embedding",
                    query_vector=query_vector,
                    distance_measure=DistanceMeasure.COSINE,
                    limit=top_k
                )
                .get()
            )
            
            formatted_results = []
            for i, doc in enumerate(results):
                data = doc.to_dict()
                metadata = data.get("metadata", {})
                metadata["text"] = data.get("text", "") # Ensure text is in metadata for Search integration
                
                formatted_results.append({
                    "index": i,
                    "distance": 0.0,  # Cosine distance not explicitly exposed as property in Python SDK snapshot directly
                    "metadata": metadata
                })
            return formatted_results

        except Exception as e:
            print(f"[WARNING] Firestore native vector search failed: {e}")
            self.native_search_failed = True  # Flag to bypass native search next time
            return self._local_search(query_embedding, top_k)

    def _local_search(self, query_embedding: List[float], top_k: int = 5):
        try:
            # Check if we have a populated local cache
            if self.local_cache is None:
                print("[INFO] Local cache is empty. Retrieving all documents from Firestore...")
                docs = self.collection.get()
                all_chunks = []
                for doc in docs:
                    data = doc.to_dict()
                    embedding_vector = data.get("embedding")
                    if embedding_vector:
                        all_chunks.append({
                            "id": doc.id,
                            "text": data.get("text", ""),
                            "metadata": data.get("metadata", {}),
                            "embedding": np.array(embedding_vector, dtype=np.float32)
                        })
                self.local_cache = all_chunks
                print(f"[INFO] Successfully cached {len(self.local_cache)} documents from Firestore.")
            else:
                print(f"[INFO] Using {len(self.local_cache)} cached documents for local search.")

            if not self.local_cache:
                print("[WARNING] No documents found in Firestore collection.")
                return []
            
            # Compute cosine similarities via vectorized matrix operation (extremely fast!)
            query_arr = np.array(query_embedding, dtype=np.float32)
            query_norm = np.linalg.norm(query_arr)
            
            if query_norm == 0:
                return []
                
            embeddings_matrix = np.array([chunk["embedding"] for chunk in self.local_cache])
            norms = np.linalg.norm(embeddings_matrix, axis=1)
            
            dots = np.dot(embeddings_matrix, query_arr)
            similarities = dots / (norms * query_norm + 1e-8)
            
            # Get the top K indexes
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            formatted_results = []
            for score_idx, idx in enumerate(top_indices):
                chunk = self.local_cache[idx]
                score = similarities[idx]
                metadata = dict(chunk["metadata"])
                metadata["text"] = chunk["text"]
                formatted_results.append({
                    "index": score_idx,
                    "distance": float(score),
                    "metadata": metadata
                })
            
            print(f"[INFO] Local search fallback successfully retrieved {len(formatted_results)} results.")
            return formatted_results
        except Exception as fallback_err:
            print(f"[ERROR] Local search fallback failed: {fallback_err}")
            raise fallback_err

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying vector store for: '{query_text}'")
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            
        query_emb = self.model.encode([query_text])[0]
        return self.search(query_emb.tolist(), top_k=top_k)

# Example usage
if __name__ == "__main__":
    from src.data_loader import load_all_documents
    docs = load_all_documents("data")
    store = FirestoreVectorStore()
    store.build_from_documents(docs)
    print("Store built successfully!")