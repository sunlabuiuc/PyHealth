from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
import pickle
import os
from langchain.embeddings import OllamaEmbeddings

def vectorize_corpus(corpus_path):
    print(f"Vectorizing {corpus_path}...")
    loader = UnstructuredFileLoader(corpus_path)
    raw_documents = loader.load()

    print("Splitting text...")
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=600,
        chunk_overlap=100,
        length_function=len,
    )
   
    # Create the document embeddings
    documents = text_splitter.split_documents(raw_documents)
    with open("docs.pkl", "wb") as f:
        pickle.dump(documents, f)


vectorize_corpus('corpus/pyhealth-text.txt')
vectorize_corpus('corpus/pyhealth-code.txt')