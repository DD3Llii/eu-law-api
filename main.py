from fastapi import FastAPI
import pickle
import numpy as np
import requests
import faiss

app = FastAPI()

# Jouw data laden
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)
    chunks = data['chunks']
    index = data['index']

# De externe AI-motor (gratis van Hugging Face)
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"

@app.get("/")
def home():
    return {"status": "Online via External AI Engine"}

@app.get("/ask")
def ask_law(query: str):
    # Vraag versturen naar de externe motor
    response = requests.post(API_URL, json={"inputs": query})
    query_vector = response.json()
    
    # Zoeken in je 836 blokken
    D, I = index.search(np.array([query_vector]).astype('float32'), k=3)
    answers = [chunks[i] for i in I[0]]
    
    return {"query": query, "top_answers": answers}
