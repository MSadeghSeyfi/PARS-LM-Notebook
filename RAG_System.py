from langchain_community.vectorstores import Chroma
from JinaStep import JinaPersianChunker

class PersianRAGSystem:
    def __init__(self, jina_api_key):
        self.chunker = JinaPersianChunker(jina_api_key)
        self.vectorstore = Chroma(
            embedding_function=self.chunker.embeddings,
            persist_directory="./chroma_db"
        )
    
    def add_documents(self, pdf_texts):
        all_chunks = []
        for text in pdf_texts:
            chunks = self.chunker.create_chunks(text)
            all_chunks.extend(chunks)
        
        # Add to vector store
        self.vectorstore.add_texts(all_chunks)
        
    def query(self, question, k=5):
        # Search با query-specific embedding
        docs = self.vectorstore.similarity_search(question, k=k)
        return docs