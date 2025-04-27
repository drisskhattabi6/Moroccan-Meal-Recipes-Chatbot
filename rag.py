import json, re
import faiss
import pickle
import socket
import logging
import time, os
import requests
import numpy as np
from nltk import download
from nltk.data import find
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.cache import BaseCache
from langchain_core.callbacks import Callbacks
from duckduckgo_search import DDGS

load_dotenv()

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

OllamaLLM.model_rebuild()

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

def get_recipe_images(query, max_results=3):
    with DDGS() as ddgs:
        results = ddgs.images(keywords=query, max_results=max_results)
        return [result['image'] for result in results]

def is_connected():
    try:
        # Connect to the host -- tells us if the host is actually reachable
        socket.create_connection(("www.google.com", 80))
        return True
    except OSError:
        pass
    return False

class RAGSystem :
    def __init__(
            self, collection_name: str, 
            db_path: str ="Moroccan_Recipes_DB",
            n_results: int =5,
            task_code : int =1
        ) :

        self.collection_name = collection_name
        self.db_path = db_path
        self.n_results = n_results
        self.task_code = task_code

        if not self.collection_name:
            raise ValueError("'collection_name' parameter is required.")

        self.logger = self._setup_logging()
        self.embedding_model = OllamaEmbeddings(model="mxbai-embed-large:latest")

        self.index_path = os.path.join(db_path, f"{collection_name}.index")
        self.meta_path = os.path.join(db_path, f"{collection_name}.pkl")

        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)

        if os.path.exists(self.meta_path):
            with open(self.meta_path, 'rb') as f:
                self.collection = pickle.load(f)
    
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

        index = self.index
        _, I = index.search(np.array([embedding]), n_results)
        return [self.collection[i] for i in I[0]] if I.any() else []
    
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
    
    def _get_prompt1(self, query, context) :
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
        - if there are multiple answers, provide the most relevant one.
        - If there is a list of ingredients, provide it as a structured list.
        - If there is a list of preparation steps, provide it as a structured list.
        - always provide the recipe name, description, ingredients, and preparation steps if available.

    ### **Answer:**

    """
        return prompt
    
    def _get_prompt2(self, query, context) :
        prompt = f"""
    You are an AI assistant specialized for Recommend Meal depends on User Ingredients or Description based **only** on the provided context. 
    The context is about Moroccan Recipes and Meals.
    The context contains many recipes and meals related to the user question, and it's structured with sections separated by `########`. 

    ### **Context:**  
        '''  
        {context}  
        '''  

    ### **User Ingredients or Description:**  
        "{query}"  

    ### **Instructions:**  
        - Answer concisely and accurately using only the given context.  
        - Put what you find from the context **without summarizing**.
        - Answer directly and concisely.
        - Recommend 3 Meal(s) depends on User Ingredients or Description.
        - always provide the recipe name, description, ingredients, and preparation steps (if available) in structured unordered list.

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

        images = []
        if is_connected():
            text = reranked_retrieved_docs[0]
            match = re.search(r'Recipe Name\s*:\s*(.*)', text)
            if match:
                recipe_name = match.group(1)
                images = get_recipe_images(recipe_name)
            else:
                images = get_recipe_images(query)

        context = "\n\n########\n\n".join(reranked_retrieved_docs)
        
        if self.task_code == 1 :
            prompt = self._get_prompt1(query, context)
        else : 
            prompt = self._get_prompt2(query, context)

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
            'response_time': self._format_time(response_time),
            'images': images,
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

        images = []
        if is_connected():
            if self.task_code == 1:
                match = re.search(r'Recipe Name\s*:\s*(.*)', reranked_retrieved_docs[0])
                if match:
                    recipe_name = match.group(1)
                    images = get_recipe_images(recipe_name)
                else:
                    images = get_recipe_images(query)
            
            else : 
                for i in range(3) :
                    match = re.search(r'Recipe Name\s*:\s*(.*)', reranked_retrieved_docs[i])
                    if match:
                        recipe_name = match.group(1)
                        images.append(get_recipe_images(recipe_name, max_results=1))
                    else:
                        images.append(get_recipe_images(query, max_results=1))


        context = "\n\n########\n\n".join(reranked_retrieved_docs)

        if self.task_code == 1 :
            prompt = self._get_prompt1(query, context)
        else : 
            prompt = self._get_prompt2(query, context)

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": llm_name,
                "messages": [{
                    "role": "user",
                    "content": prompt
                }],
            })
        )

        response_data = json.loads(response.text)
        print(response_data)

        # Check if there's an error key in the response
        if "error" in response_data:
            error_message = response_data["error"].get("message", "Unknown error")
            raise Exception(f"API Error: {error_message}")
        
        return response_data["choices"][0]["message"]["content"], response_data["usage"]["total_tokens"], images


if __name__ == "__main__":
    images = get_recipe_images("Rfissa Moroccan recipe")
    print(images)
