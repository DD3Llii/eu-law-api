import os, pickle, zipfile, numpy as np
from fastapi import FastAPI, HTTPException
from google import genai

app = FastAPI(title="Multi-Law Legal AI - Fixed")

# Config
API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)

def load_full_database():
    all_chunks = []
    all_embeddings = []
    
    # Laden van beide gecomprimeerde delen
    for filename in ['data1.zip', 'data2.zip']:
        if os.path.exists(filename):
            with zipfile.ZipFile(filename, 'r') as z:
                # We openen de pkl binnen de zip
                pkl_name = filename.replace('.zip', '.pkl')
                with z.open(pkl_name) as f:
                    data = pickle.load(f)
                    all_chunks.extend(data['chunks'])
                    all_embeddings.append(data['embeddings'])
    
    if not all_chunks:
        return [], None
        
    # Plak de embeddings aan elkaar en zet om naar float32 voor berekeningen
    return all_chunks, np.vstack(all_embeddings).astype('float32')

chunks, embeddings = load_full_database()

@app.get("/")
def home():
    return {"status": "Online", "database_size": len(chunks)}

@app.get("/ask")
def ask(query: str):
    if not query or not chunks:
        raise HTTPException(status_code=400, detail="Systeem niet gereed of geen vraag.")

    try:
        # Stap 1: Vraag omzetten naar vector
        res = client.models.embed_content(
            model="text-embedding-004",
            contents=query,
            config={'task_type': 'RETRIEVAL_QUERY'}
        )
        query_vector = np.array(res.embeddings[0].values).astype('float32')
        
        # Stap 2: Zoeken (Afstand berekenen)
        distances = np.linalg.norm(embeddings - query_vector, axis=1)
        
        # FIX: Hier stond eerst 'indices' in plaats van 'top_indices'
        top_indices = np.argsort(distances)[:5]
        
        return {
            "query": query, 
            "results": [chunks[i] for i in top_indices]
        }
    except Exception as e:
        return {"error": "API Fout", "details": str(e)}
