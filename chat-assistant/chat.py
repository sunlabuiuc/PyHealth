import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model_base_url = "http://localhost:11501"

# Load your saved data
with open("docs.pkl", "rb") as f:
    docs = pickle.load(f)  # Should be a list of strings or dicts

# Detect and convert LangChain-style Document objects
if hasattr(docs[0], "page_content"):
    texts = [doc.page_content for doc in docs]
elif isinstance(docs[0], dict):
    texts = [doc["text"] for doc in docs]
else:
    texts = docs  # assume already a list of strings


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedding_model.encode(texts, show_progress_bar=True)

index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))

def retrieve(query, top_k=3):
    query_embedding = embedding_model.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)
    results = [texts[i] for i in I[0]]

    # Ensure results are all strings (in case still Document objects)
    return [
        r.page_content if hasattr(r, "page_content") else str(r)
        for r in results
    ]


def generate_with_ollama(prompt, model="llama3"):
    try:
        res = requests.post(
            model_base_url,
            json={"model": model, "prompt": prompt, "stream": False}
        )
        return res.json().get("response", "[No response]")
    except Exception as e:
        return f"[Error contacting Ollama]: {e}"

def rag_ask(query):
    context = retrieve(query, top_k=3)
    context_text = "\n".join(context)
    # Adjust prompt if needed
    prompt = f"""Answer the question using the context below. If the answer is not in the context, say you don't know.

Context:
{context_text}

Question: {query}
Answer:"""
    return generate_with_ollama(prompt)

# Streamlit UI
st.title("🧠 Local RAG Chat (FAISS + Ollama)")
if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Ask a question", key="input")
if user_input:
    response = rag_ask(user_input)
    st.session_state.history.append((user_input, response))

for q, a in reversed(st.session_state.history):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Assistant:** {a}")
