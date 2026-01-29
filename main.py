from fastapi import FastAPI
import pickle
import numpy as np
import requests
import os

app = FastAPI()

# Laad de data (nu zonder FAISS objecten)
with open('data_ai_act.pkl', 'rb') as f:
    data = pickle.load(f)
    chunks = data['chunks']
    embeddings = np.array(data['embeddings']).astype('float32')

API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"

@app.get("/")
def home():
    return {"status": "AI Act Service Online (Numpy Mode)"}

@app.get("/ask")
def ask_law(query: str):
    # 1. Haal vector op bij Hugging Face
    response = requests.post(API_URL, json={"inputs": query})
    query_vector = np.array(response.json()).astype('float32')
    
    # 2. Zoek met Numpy (Euclidean distance)
    # Dit vervangt de FAISS-index en is superstabiel
    distances = np.linalg.norm(embeddings - query_vector, axis=1)
    top_indices = np.argsort(distances)[:3]
    
    answers = [chunks[i] for i in top_indices]
    return {"query": query, "top_answers": answers}
