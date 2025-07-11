from langchain_community.embeddings import JinaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests

class JinaPersianChunker:
    def __init__(self, jina_api_key):
        # Jina v3 optimized settings
        self.embeddings = JinaEmbeddings(
            jina_api_key=jina_api_key,
            model_name="jina-embeddings-v3",
            # بهترین تنظیمات برای multilingual
            task="retrieval.passage",  # برای documents
            dimensions=1024,  # max dimension برای بهترین quality
            late_chunking=True,  # بهبود accuracy برای long texts
            embedding_type="float"
        )
        
        self.query_embeddings = JinaEmbeddings(
            jina_api_key=jina_api_key,
            model_name="jina-embeddings-v3", 
            task="retrieval.query",  # برای queries
            dimensions=1024,
            embedding_type="float"
        )
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,  # optimal برای Jina v3
            chunk_overlap=80,  # 20% overlap
            separators=[
                '\n\n',      # پاراگراف
                '\n',        # خط جدید
                '؟',         # سوال فارسی
                '!',         # تعجب
                '.',         # نقطه انگلیسی
                '۔',         # نقطه عربی/اردو
                '؛',         # نقطه‌ویرگول فارسی
                '،',         # ویرگول فارسی
                ',',         # ویرگول انگلیسی
                ' '          # فاصله
            ]
        )

    def create_chunks(self, text):
        chunks = self.splitter.split_text(text)
        # فیلتر chunks کوتاه
        filtered_chunks = [chunk for chunk in chunks if len(chunk.strip()) > 40]
        return filtered_chunks

    def embed_documents(self, texts):
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, query):
        return self.query_embeddings.embed_query(query)