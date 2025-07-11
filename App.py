import streamlit as st
from langchain_community.document_loaders import PyPDFium2Loader
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import re

class App:
    def __init__(self):
        self.translations = {
            'fa' : {
                'change_language': 'تغییر زبان',
                'current_language': 'زبان فعلی: فارسی',
                'app_title': 'کار با PDF',
                'placeholder_genre': 'لطفا در باره ی PDF سوالی بپرسید .',
                'select_file' : 'يک فايل PDf برگزينيد .'
            },
            'en' : {
                'change_language': 'Change Language',
                'current_language': 'Current Language: English',
                'app_title': 'work with PDF',
                'placeholder_genre': 'Please ask a question about pdf .',
                'select_file' : 'Please select a PDF file .'
            }
        }

        if 'language' not in st.session_state:
            st.session_state.language = 'fa'

        if st.button(self.translations[st.session_state.language]['change_language']):
            st.session_state.language = 'en' if st.session_state.language == 'fa' else "fa"

        self.language = st.session_state.language
        self.text_align, self.direction = self.get_text_alignment()
        self.inject_css()
        self.model = None

    def get_text_alignment(self):
            if self.language == 'fa':
                return 'right', 'rtl'
            else:
                return 'left', 'ltr'    
        
    def inject_css(self):
            st.markdown(f"""
                <style>
                        
                    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500&display=swap');
                    
                    * {{
                        font-family: 'Vazirmatn', sans-serif;
                        direction: {self.direction};
                        text-align: {self.text_align};
                    }}

                </style>
            """, unsafe_allow_html=True)  

    def minimal_persian_fix(self, text):
                # Ascending
                # تصحیح کاراکترهای اشتباه
                char_fixes = {
                    'ͷ': 'ک',  # کاراکتر اشتباه به جای ک
                    'ͺ': 'ک',  # کاراکتر اشتباه دیگر
                    'ͽ': 'گ',  # کاراکتر اشتباه به جای گ
                    'ؤ': 'و',  # و همزه‌دار عربی به فارسی
                    'ى': 'ی',  # ی عربی به فارسی
                    'ة': 'ه',  # ته مربوطه عربی به فارسی
                    'ك': 'ک',  # کاف عربی به فارسی
                    'ي': 'ی',  # یا عربی به فارسی
                }

                # جایگزینی کاراکترها
                for wrong, correct in char_fixes.items():
                    text = text.replace(wrong, correct)

                # تصحیح مشکل خاص chr(876) + فاصله
                text = text.replace(chr(876) + ' ', 'ی ')
                text = text.replace(chr(876) + chr(13), 'ی ')
                text = text.replace(chr(876), 'ی')  # برای موارد دیگر chr(876)

                return text

    def extract_text_from_pdf(self, pdf_path: str) -> str:
            try:
                # Load the PDF using PyPDFium2Loader
                loader = PyPDFium2Loader(pdf_path)
                documents = loader.load()
                text = ""

                # Iterate through all documents (each represents a page)
                for doc in documents:
                    original_content = doc.page_content
                    corrected_content = self.minimal_persian_fix(original_content)
                    text += corrected_content + "\n"  # Add newline between pages
                
                return text
            except Exception as e:
                return f"Error extracting text: {str(e)}"
        

    def advanced_persian_chunking(self, text: str) -> List[str]:
        """
        پیاده‌سازی پیشرفته تقسیم متن فارسی به چانک‌های بهینه
        """
        
        # 1. پیش‌پردازش اولیه متن
        def preprocess_text(text: str) -> str:
            # حذف فاصله‌های اضافی
            text = re.sub(r'\s+', ' ', text)
            # تنظیم فاصله‌ها قبل و بعد علائم نگارشی فارسی
            text = re.sub(r'\s*([.!?؟:؛،])\s*', r'\1 ', text)
            # حذف خطوط خالی اضافی
            text = re.sub(r'\n\s*\n', '\n\n', text)
            return text.strip()
        
        # 2. تعریف جداکننده‌های بهینه برای فارسی
        persian_separators = [
            "\n\n\n",          # پاراگراف‌های جدا
            "\n\n",            # پاراگراف‌ها
            "\n",              # خطوط جدید
            "؟",               # علامت سوال فارسی
            "!",               # علامت تعجب
            ".",               # نقطه
            "؛",               # نقطه ویرگول فارسی
            ":",               # دو نقطه
            "،",               # ویرگول فارسی
            ",",               # ویرگول انگلیسی
            " و ",             # حرف ربط فارسی
            " که ",            # کلمات ربط فارسی
            " در ",
            " از ",
            " به ",
            " با ",
            " ",               # فاصله
            ""                 # کاراکتر خالی
        ]
        
        # 3. تنظیمات بهینه برای متن فارسی
        chunk_size = 800       # اندازه مناسب برای فارسی
        chunk_overlap = 150    # overlap کافی برای حفظ context
        
        # 4. ایجاد text splitter بهینه
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=persian_separators,
            keep_separator=True,    # حفظ جداکننده‌ها
            add_start_index=True,   # اضافه کردن ایندکس شروع
        )
        
        # 5. پیش‌پردازش متن
        processed_text = preprocess_text(text)
        
        # 6. تقسیم به چانک‌ها
        chunks = text_splitter.split_text(processed_text)
        
        # 7. پس‌پردازش چانک‌ها
        def postprocess_chunk(chunk: str) -> str:
            # حذف فاصله‌های ابتدا و انتها
            chunk = chunk.strip()
            # اطمینان از وجود حداقل طول
            if len(chunk) < 50:
                return None
            # اضافه کردن نقطه در صورت نیاز
            if not chunk.endswith(('.', '!', '؟', ':', '؛')):
                chunk += '.'
            return chunk
        
        # 8. فیلتر و بهینه‌سازی چانک‌ها
        final_chunks = []
        for chunk in chunks:
            processed_chunk = postprocess_chunk(chunk)
            if processed_chunk:
                final_chunks.append(processed_chunk)
        
        # 9. اطلاعات آماری
        st.write(f"📊 آمار تقسیم‌بندی:")
        st.write(f"   • تعداد کل چانک‌ها: {len(final_chunks)}")
        st.write(f"   • متوسط طول چانک: {sum(len(chunk) for chunk in final_chunks) // len(final_chunks) if final_chunks else 0} کاراکتر")
        st.write(f"   • کوتاه‌ترین چانک: {min(len(chunk) for chunk in final_chunks) if final_chunks else 0} کاراکتر")
        st.write(f"   • بلندترین چانک: {max(len(chunk) for chunk in final_chunks) if final_chunks else 0} کاراکتر")
        
        return final_chunks

    # 10. تابع اضافی برای بررسی کیفیت چانک‌ها
    def analyze_chunks_quality(self, chunks: List[str]) -> dict:
        """تحلیل کیفیت چانک‌های تولید شده"""
        
        analysis = {
            'total_chunks': len(chunks),
            'avg_length': sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0,
            'min_length': min(len(chunk) for chunk in chunks) if chunks else 0,
            'max_length': max(len(chunk) for chunk in chunks) if chunks else 0,
            'empty_chunks': sum(1 for chunk in chunks if len(chunk.strip()) == 0),
            'short_chunks': sum(1 for chunk in chunks if len(chunk) < 100),
            'optimal_chunks': sum(1 for chunk in chunks if 200 <= len(chunk) <= 1000),
        }
        
        return analysis            
                
    def display_app(self):
        st.set_page_config(page_title="Persian NotebookLM 📚", page_icon= "content/PARS-LM-NOTEBOOK.png")
        st.title("Persian NotebookLM 📚")
        st.write(self.translations[self.language]['current_language'])
        st.header(self.translations[self.language]['app_title'], divider="red") 
        uploaded_file = st.file_uploader(f"{self.translations[self.language]['select_file']} PDF", type=["pdf"])
        
        if uploaded_file is not None:
            # Save the uploaded file temporarily
            file_path = Path("temp.pdf")
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            extracted_text = self.extract_text_from_pdf(file_path)
            
            # ✅ مرحله جدید: تقسیم متن به چانک‌ها
            progress_bar.progress(50)
            status_text.text("در حال تقسیم متن به بخش‌های کوچکتر...")

            #  # Display extracted text
            st.subheader("متن استخراج‌شده:")
            st.text_area("متن", extracted_text, height=400)

            text_chunks = self.advanced_persian_chunking(extracted_text)

            st.write(text_chunks)
            
            # Clean up temporary file
            file_path.unlink()                     

if __name__ == "__main__":
    app = App()
    app.display_app()