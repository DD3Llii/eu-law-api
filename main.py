from fastapi import FastAPI
import pickle
import numpy as np
import requests

app = FastAPI()

# Milieudata laden (data.pkl moet op GitHub staan!)
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)
    chunks = data['chunks']
    embeddings = np.array(data['embeddings']).astype('float32')

# De allernieuwste URL voor de AI-motor
API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"

@app.get("/")
def home():
    return {"status": "Milieuwet API is Online"}

@app.get("/ask")
def ask_law(query: str):
    # Stap 1: Vraag stellen aan de AI-motor
    response = requests.post(API_URL, json={"inputs": query})
    result = response.json()
    
    # Stap 2: Check of de AI-motor nog aan het opstarten is
    if isinstance(result, dict) and "error" in result:
        return {"query": query, "top_answers": ["AI-motor start op. Probeer over 15 seconden opnieuw."], "debug": result}
    
    # Stap 3: Bereken het antwoord met Numpy
    query_vector = np.array(result).astype('float32')
    distances = np.linalg.norm(embeddings - query_vector, axis=1)
    top_indices = np.argsort(distances)[:3]
    
    answers = [chunks[i] for i in top_indices]
    return {"query": query, "top_answers": answers}
