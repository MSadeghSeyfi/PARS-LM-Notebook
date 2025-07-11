from groq import Groq
import os

def test_groq_qwen():
    """تست API شرکت Groq با مدل qwen3-32b"""
    
    # تنظیم API Key
    API_KEY = "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"  # جایگزین با کلید واقعی
    
    # ایجاد کلاینت Groq
    client = Groq(api_key=API_KEY)
    
    # تست‌های مختلف فارسی
    test_messages = [
        {
            "name": "تست پاسخ ساده فارسی",
            "prompt": "سلام! لطفاً خودت را معرفی کن و بگو چه کاری می‌توانی انجام دهی؟"
        },
        {
            "name": "تست تحلیل متن فارسی", 
            "prompt": "این جمله را تحلیل کن: 'هوش مصنوعی آینده فناوری است.' نکات کلیدی چیست؟"
        },
        {
            "name": "تست ترجمه",
            "prompt": "این متن انگلیسی را به فارسی ترجمه کن: 'Artificial Intelligence is transforming the world'"
        },
        {
            "name": "تست خلاصه‌سازی",
            "prompt": "این متن را خلاصه کن: 'تکنولوژی هوش مصنوعی در سال‌های اخیر پیشرفت‌های چشمگیری داشته است. از پردازش زبان طبیعی گرفته تا بینایی کامپیوتر، این فناوری در حال تغییر دادن نحوه زندگی ما است.'"
        }
    ]
    
    print("🚀 شروع تست API Groq با مدل qwen3-32b\n")
    print("="*50)
    
    for i, test in enumerate(test_messages, 1):
        try:
            print(f"\n📝 تست {i}: {test['name']}")
            print(f"سوال: {test['prompt']}")
            print("-" * 30)
            
            # ارسال درخواست به API
            response = client.chat.completions.create(
                model="qwen/qwen3-32b",  # مدل مورد نظر
                messages=[
                    {
                        "role": "system", 
                        "content": "تو یک دستیار هوشمند هستی که به زبان فارسی پاسخ می‌دهی. پاسخ‌هایت دقیق، مفید و مفصل باشند."
                    },
                    {
                        "role": "user", 
                        "content": test['prompt']
                    }
                ],
                temperature=0.7,
                max_tokens=1000,
                top_p=1,
                stream=False
            )
            
            # نمایش پاسخ
            answer = response.choices[0].message.content
            print(f"✅ پاسخ: {answer}")
            
            # اطلاعات اضافی
            print(f"📊 تعداد توکن‌های استفاده شده: {response.usage.total_tokens}")
            print(f"⏱️ مدت زمان: {response.usage}")
            
        except Exception as e:
            print(f"❌ خطا در تست {i}: {str(e)}")
        
        print("="*50)
    
    print("\n🎯 تست‌ها تمام شد!")

def simple_chat_test():
    """تست ساده چت با مدل"""
    
    API_KEY = "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"
    client = Groq(api_key=API_KEY)
    
    print("💬 تست چت ساده - سوال خود را بپرسید:")
    
    while True:
        user_input = input("\nشما: ")
        if user_input.lower() in ['exit', 'خروج', 'quit']:
            break
            
        try:
            response = client.chat.completions.create(
                model="qwen/qwen3-32b",
                messages=[
                    {"role": "user", "content": user_input}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            print(f"🤖 مدل: {response.choices[0].message.content}")
            
        except Exception as e:
            print(f"❌ خطا: {e}")

if __name__ == "__main__":
    # انتخاب نوع تست
    print("انتخاب کنید:")
    print("1. تست کامل")
    print("2. تست چت ساده")
    
    choice = input("انتخاب (1 یا 2): ")
    
    if choice == "1":
        test_groq_qwen()
    elif choice == "2":
        simple_chat_test()
    else:
        print("انتخاب نامعتبر!")