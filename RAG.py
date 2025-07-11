import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from typing import List, Dict, Any
import json

class PersianRAGSystem:
    """سیستم RAG بهینه شده برای زبان فارسی"""
    
    def __init__(self, jina_api_key: str = None):
        """مقداردهی اولیه سیستم RAG"""
        
        # بارگذاری مدل embedding
        self.embedding_model = SentenceTransformer('jinaai/jina-embeddings-v3', 
                                                   trust_remote_code=True)
        
        # تنظیمات مدل برای چندزبانه بودن
        self.embedding_model.max_seq_length = 8192  # طول بیشتر برای متون فارسی
        
        # متغیرهای ذخیره‌سازی
        self.chunks = []           # لیست چانک‌های متنی
        self.embeddings = None     # ماتریس embeddings
        self.vector_index = None   # ایندکس FAISS
        self.chunk_metadata = []   # اطلاعات اضافی چانک‌ها
        
        print("✅ سیستم RAG با موفقیت مقداردهی شد")
        print(f"📊 مدل embedding: jina-embeddings-v3")
        print(f"🌐 پشتیبانی چندزبانه: فارسی، انگلیسی، عربی")

    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """تبدیل چانک‌ها به بردارهای embedding"""
        
        print(f"🔄 در حال تولید embeddings برای {len(chunks)} چانک...")
        
        # تولید embeddings
        embeddings = self.embedding_model.encode(
            chunks,
            batch_size=8,           # batch کوچک برای مدیریت حافظه
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # نرمال‌سازی برای بهبود جستجو
        )
        
        print(f"✅ تولید embeddings با موفقیت انجام شد")
        print(f"📐 ابعاد هر embedding: {embeddings.shape[1]}")
        
        return embeddings

    def build_vector_index(self, embeddings: np.ndarray):
        """ساخت ایندکس FAISS برای جستجوی سریع"""
        
        print("🏗️ در حال ساخت ایندکس برداری...")
        
        # تعیین نوع ایندکس بر اساس تعداد چانک‌ها
        dimension = embeddings.shape[1]
        
        if len(embeddings) < 1000:
            # برای داده‌های کم: استفاده از IndexFlatIP (دقیق‌ترین)
            self.vector_index = faiss.IndexFlatIP(dimension)
        else:
            # برای داده‌های زیاد: استفاده از IndexIVFFlat (سریع‌تر)
            nlist = min(100, len(embeddings) // 10)
            quantizer = faiss.IndexFlatIP(dimension)
            self.vector_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            self.vector_index.train(embeddings.astype('float32'))
        
        # اضافه کردن embeddings به ایندکس
        self.vector_index.add(embeddings.astype('float32'))
        
        print(f"✅ ایندکس برداری ساخته شد: {self.vector_index.ntotal} بردار")

    def add_documents(self, chunks: List[str]):
        """اضافه کردن اسناد به سیستم RAG"""
        
        # ذخیره چانک‌ها
        self.chunks = chunks
        
        # ایجاد metadata برای هر چانک
        self.chunk_metadata = [
            {
                'chunk_id': i,
                'length': len(chunk),
                'preview': chunk[:100] + "..." if len(chunk) > 100 else chunk
            }
            for i, chunk in enumerate(chunks)
        ]
        
        # تولید embeddings
        self.embeddings = self.create_embeddings(chunks)
        
        # ساخت ایندکس برداری
        self.build_vector_index(self.embeddings)
        
        print(f"🎯 {len(chunks)} سند با موفقیت به سیستم اضافه شد")

    def search_similar_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """جستجوی چانک‌های مشابه برای سوال"""
        
        if self.vector_index is None:
            raise ValueError("❌ ابتدا باید اسناد را به سیستم اضافه کنید")
        
        print(f"🔍 جستجو برای: {query[:50]}...")
        
        # تبدیل سوال به embedding
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # جستجو در ایندکس
        scores, indices = self.vector_index.search(
            query_embedding.astype('float32'), 
            top_k
        )
        
        # آماده‌سازی نتایج
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0:  # اطمینان از وجود ایندکس
                results.append({
                    'rank': i + 1,
                    'chunk_id': idx,
                    'similarity_score': float(score),
                    'text': self.chunks[idx],
                    'metadata': self.chunk_metadata[idx]
                })
        
        print(f"✅ {len(results)} نتیجه مرتبط یافت شد")
        return results

    def save_system(self, filepath: str):
        """ذخیره سیستم RAG برای استفاده مجدد"""
        
        system_data = {
            'chunks': self.chunks,
            'embeddings': self.embeddings.tolist() if self.embeddings is not None else None,
            'chunk_metadata': self.chunk_metadata
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(system_data, f)
        
        # ذخیره ایندکس FAISS
        if self.vector_index is not None:
            faiss.write_index(self.vector_index, filepath.replace('.pkl', '.faiss'))
        
        print(f"💾 سیستم در {filepath} ذخیره شد")

    def load_system(self, filepath: str):
        """بارگذاری سیستم RAG ذخیره شده"""
        
        with open(filepath, 'rb') as f:
            system_data = pickle.load(f)
        
        self.chunks = system_data['chunks']
        self.chunk_metadata = system_data['chunk_metadata']
        
        if system_data['embeddings']:
            self.embeddings = np.array(system_data['embeddings'])
            
        # بارگذاری ایندکس FAISS
        faiss_path = filepath.replace('.pkl', '.faiss')
        try:
            self.vector_index = faiss.read_index(faiss_path)
            print(f"📂 سیستم از {filepath} بارگذاری شد")
        except:
            print("⚠️ فایل ایندکس یافت نشد، ایندکس مجدد ساخته می‌شود...")
            if self.embeddings is not None:
                self.build_vector_index(self.embeddings)