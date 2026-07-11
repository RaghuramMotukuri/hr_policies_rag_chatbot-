import os
from dotenv import load_dotenv

load_dotenv()

class RAGSearch:
    def __init__(self, collection_name: str = "hr_policies_embeddings", llm_model: str = "llama-3.3-70b-versatile", vectorstore = None):
        if vectorstore is not None:
            self.vectorstore = vectorstore
        else:
            from src.vectorstore import FirestoreVectorStore
            self.vectorstore = FirestoreVectorStore(collection_name=collection_name)
        
        self.llm_model = llm_model
        self.llm = None

    def _get_llm(self):
        if self.llm is None:
            # Load Groq API key from environment variable
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                print("[WARNING] GROQ_API_KEY is not set in the environment variables.")
                
            from langchain_groq import ChatGroq
            self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=self.llm_model)
            print(f"[INFO] Groq LLM initialized: {self.llm_model}")
        return self.llm

    def search_and_summarize(self, query: str, chat_history: list = None, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        
        history_str = ""
        if chat_history:
            formatted_history = []
            for msg in chat_history[-6:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                formatted_history.append(f"{role}: {msg['content']}")
            history_str = "\n".join(formatted_history)

        prompt = f"""You are a professional HR Assistant chatbot. Answer the user's query using the provided context and the preceding conversation history.
        
Guidelines for your response:
1. STRICT FORMAT COMPLIANCE: If the user explicitly asks for the response in a specific format (such as JSON, CSV, HTML, raw text, a single sentence, bullet points only, etc.), you MUST strictly output the response in that exact format.
2. DEFAULT SEMANTIC FORMATTING: If the user does not request a specific format, structure your response in a clean, short, SEMANTIC MARKDOWN format. Keep the reply simple, clear, and concise (1-2 brief paragraphs or bullet lists maximum):
   - Use clear bold headings (###) for distinct sections.
   - Present lists, rules, or key steps using clean bullet points.
   - Use Markdown Tables for numerical comparisons, structures, or matrix values (like PTO accrual grids).
   - Use blockquotes (e.g., > **Note:** ...) for warnings or crucial guidelines.
3. Clearly cite the source document name and page number at the end of sections where you retrieved information from.
4. If the context does not contain enough information to answer the query, state that clearly in a clean, polite paragraph. Do not invent details.

Conversation History:
{history_str}

Current Query: {query}

Context:
{context}

Response:"""
        response = self._get_llm().invoke([prompt])
        return response.content

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)