from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
import os
from env import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def vectorize_corpus(corpus_path):
    print(f"Vectorizing {corpus_path}...")
    # print(f"Loading data {corpus_path}...")
    loader = UnstructuredFileLoader(corpus_path)
    raw_documents = loader.load()


    # print("Splitting text...")
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=600,
        chunk_overlap=100,
        length_function=len,
    )
    documents = text_splitter.split_documents(raw_documents)

    # print("Creating vectorstore...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    base_name, ext = os.path.splitext(corpus_path)
    vectorstore_path = base_name + '.pkl'
    with open(vectorstore_path, "wb") as f:
        pickle.dump(vectorstore, f)


vectorize_corpus('corpus/pyhealth-text.txt')
vectorize_corpus('corpus/pyhealth-code.txt')