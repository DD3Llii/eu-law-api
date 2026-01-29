from fastapi import FastAPI
import pickle
import numpy as np
import requests

app = FastAPI()

# Milieudata laden (De nieuwe data.pkl zonder FAISS)
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)
    chunks = data['chunks']
    embeddings = np.array(data['embeddings']).astype('float32')

# De externe AI-motor (gratis van Hugging Face)
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"

@app.get("/")
def home():
    return {"status": "Milieuwet API Online (Safe Mode)"}

@app.get("/ask")
def ask_law(query: str):
    # 1. Vector ophalen via internet
    response = requests.post(API_URL, json={"inputs": query})
    query_vector = np.array(response.json()).astype('float32')
    
    # 2. Zoeken in je 836 blokken met Numpy
    distances = np.linalg.norm(embeddings - query_vector, axis=1)
    top_indices = np.argsort(distances)[:3]
    
    answers = [chunks[i] for i in top_indices]
    return {"query": query, "top_answers": answers}
