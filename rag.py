import logging
import chromadb
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import time, os
import requests
import json
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# import nltk
from nltk.data import find
from nltk import download
from nltk.tokenize import word_tokenize

load_dotenv()

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

try:
    find('tokenizers/punkt')
except LookupError:
    download('punkt')

try:
    find('tokenizers/punkt_tab')
except LookupError:
    download('punkt_tab')

# Load SentenceTransformer model for semantic similarity (Contriever-style)
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

class RAGSystem :
    def __init__(
            self, collection_name: str, 
            db_path: str ="Moroccan_Recipes_ChromaDB",
            n_results: int =5
        ) :

        self.collection_name = collection_name
        self.db_path = db_path
        self.n_results = n_results

        if not self.collection_name:
            raise ValueError("'collection_name' parameter is required.")

        self.logger = self._setup_logging()
        self.embedding_model = OllamaEmbeddings(model="mxbai-embed-large:latest")
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_collection(name=self.collection_name)
        # self.logger.info("*** RAGSystem initialized ***")
    
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        return logger
    
    def _format_time(self, response_time):
        minutes = response_time // 60
        seconds = response_time % 60
        return f"{int(minutes)}m {int(seconds)}s" if minutes else f"Time: {int(seconds)}s"
    
    def _generate_embeddings(self, text: str):
        return self.embedding_model.embed_query(text)
    
    def _retrieve(self, user_text: str, n_results:int=10):
        """Retrieves relevant documents based on user input."""
        embedding = self._generate_embeddings(user_text)
        results = self.collection.query(query_embeddings=[embedding], n_results=n_results)
        
        if not results['documents']:
            return []
        
        return results['documents'][0]
    
    def _rerank_docs(self, chunks: list, query: str, top_k: int = 5):
        """
        Retrieves and ranks text chunks using BM25, semantic similarity,
        and diversity filtering inspired by ReComp.
        """
        # ----- BM25 Lexical Ranking -----
        tokenized_chunks = [word_tokenize(chunk.lower()) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        bm25_scores = bm25.get_scores(word_tokenize(query.lower()))
        
        # ----- Semantic Ranking -----
        chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True)
        query_embedding = embedder.encode([query], convert_to_tensor=True)
        semantic_scores = cosine_similarity(query_embedding, chunk_embeddings)[0]
        
        # ----- Combine Scores -----
        bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-5)
        sem_norm = (semantic_scores - np.min(semantic_scores)) / (np.max(semantic_scores) - np.min(semantic_scores) + 1e-5)
        combined_scores = 0.5 * bm25_norm + 0.5 * sem_norm

        # ----- Top-K Selection -----
        ranked_indices = np.argsort(combined_scores)[::-1]  # from best to worst

        # ----- Recomp-like Filtering (diversity-aware Top-K) -----
        final_chunks = []
        seen_embeddings = []
        i = 0

        while len(final_chunks) < top_k and i < len(ranked_indices):
            idx = ranked_indices[i]
            candidate = chunks[idx]
            candidate_emb = embedder.encode([candidate])

            is_similar = False
            for emb in seen_embeddings:
                sim = cosine_similarity(candidate_emb, emb)[0][0]
                if sim > 0.85:
                    is_similar = True
                    break

            if not is_similar:
                final_chunks.append(candidate)
                seen_embeddings.append(candidate_emb)

            i += 1

        return final_chunks
    
    def _get_prompt(self, query, context) :
        prompt = f"""
    You are an AI assistant specialized in answering questions based **only** on the provided context. 
    The context is about Moroccan Recipes and Meals.
    The context contains many recipes and meals related to the user question, and it's structured with sections separated by `########`. 

    ### **Context:**  
        '''  
        {context}  
        '''  

    ### **Question:**  
        "{query}"  

    ### **Instructions:**  
        - Answer concisely and accurately using only the given context.  
        - Put what you find from the context **without summarizing**.
        - Answer directly and concisely.

    ### **Answer:**

    """
        return prompt
    
    def generate_response(self, query: str, ollama_model):
        """Generates a response using retrieved documents and an LLM."""

        if not ollama_model :
            return "Choose and ollama LLM"

        self.logger.info(f"--> Generate Response Using Ollama LLM : {ollama_model}")

        retrieved_docs = self._retrieve(query, n_results=20)
        
        if not retrieved_docs:
            return "No relevant information found."
        
        reranked_retrieved_docs = self._rerank_docs(chunks=retrieved_docs, query=query, top_k=self.n_results)

        context = "\n########\n".join(reranked_retrieved_docs)
        
        prompt = self._get_prompt(query, context)

        self.logger.info(f"-> User Query : {query}")
        self.logger.info(f"-> Context : {prompt}")

        ollama_llm = OllamaLLM(model=ollama_model)
        
        token_count = ollama_llm.get_num_tokens(prompt)
        start_time = time.time()

        streamed_response = ""
        for chunk in ollama_llm.stream(prompt): 
            streamed_response += chunk
            yield streamed_response 

        response_time = time.time() - start_time
        self.logger.info(f"-> LLM Response : {streamed_response}")
        self.logger.info(f"-> input token count : {token_count}  |  response time : {self._format_time(response_time)}")
        metadata = {
            'token_count': token_count,
            'response_time': self._format_time(response_time)
        }
        yield metadata

    def generate_response2(self, query, llm_name='qwen/qwq-32b:free', openrouter_api_key=None) :

        if not openrouter_api_key and not OPENROUTER_API_KEY :
            return "Set OpenRouter API Key"
        
        openrouter_key = openrouter_api_key if openrouter_api_key else OPENROUTER_API_KEY

        self.logger.info(f"--> Generate Response Using OpenRouter LLM : {llm_name}")

        retrieved_docs = self._retrieve(query, n_results=20)
        
        if not retrieved_docs:
            return "No relevant information found."
        
        reranked_retrieved_docs = self._rerank_docs(chunks=retrieved_docs, query=query, top_k=self.n_results)

        context = "\n-----\n".join(reranked_retrieved_docs)

        prompt = self._get_prompt(query, context)

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": llm_name,
                "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
                ],
            })
        )

        response_data = json.loads(response.text)
        print(response_data)

        # Check if there's an error key in the response
        if "error" in response_data:
            error_message = response_data["error"].get("message", "Unknown error")
            raise Exception(f"API Error: {error_message}")
        
        return response_data["choices"][0]["message"]["content"], response_data["usage"]["total_tokens"]

    def delete_collection(self):
        self.client.delete_collection(self.collection_name)
