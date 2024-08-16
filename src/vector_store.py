import os
from langchain.docstore.document import Document
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

os.environ['PINECONE_API_KEY'] = "your_pinecone_api_key"

def create_vector_store(documents, index_name):
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    docsearch = PineconeVectorStore.from_documents(documents, embedding_function, index_name=index_name)
    return docsearch

def retrieve_vector_store(index_name):
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    docsearch = PineconeVectorStore(index_name=index_name,embedding=embedding_function)
    return docsearch