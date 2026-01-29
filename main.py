import os
import pickle
import numpy as np
from fastapi import FastAPI
from google import genai

app = FastAPI()

# 1. Google Configuratie
API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)

# 2. Data inladen
with open('data.pkl', 'rb') as f:
    db = pickle.load(f)
    chunks = db['chunks']
    embeddings = np.array(db['embeddings']).astype('float32')

@app.get("/")
def home():
    return {"status": "Online", "engine": "Google Gemini"}

@app.get("/ask")
def ask(query: str):
    # Stap 1: Vraag omzetten naar vector via Google
    res = client.models.embed_content(
        model="text-embedding-004", 
        contents=query, 
        config={'task_type': 'RETRIEVAL_QUERY'}
    )
    query_vector = np.array(res.embeddings[0].values).astype('float32')
    
    # Stap 2: Zoeken met Numpy
    distances = np.linalg.norm(embeddings - query_vector, axis=1)
    indices = np.argsort(distances)[:3]
    
    return {
        "query": query,
        "results": [chunks[i] for i in indices]
    }
