import streamlit as st
from langchain_community.document_loaders import PyPDFium2Loader
from pathlib import Path


class App:
    def __init__(self):
        st.set_page_config(page_title="Persian NotebookLM 📚", page_icon= "content/PARS-LM-NOTEBOOK.png")
        st.title("Persian NotebookLM 📚")
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

        self.language = st.session_state.language

        if st.button(self.translations[st.session_state.language]['change_language']):
            st.session_state.language = 'en' if st.session_state.language == 'fa' else "fa"
                
        st.write(self.translations[self.language]['current_language'])

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
                
    def display_app(self):
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
            
            progress_bar.progress(100)
            status_text.text("استخراج متن با موفقیت انجام شد!")
            
            #  # Display extracted text
            # st.subheader("متن استخراج‌شده:")
            # st.text_area("متن", extracted_text, height=400)
            
            # Clean up temporary file
            file_path.unlink()                     

if __name__ == "__main__":
    app = App()
    app.display_app()