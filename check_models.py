import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY not found in .env")
else:
    genai.configure(api_key=api_key)
    
    print("------------------------------------------------")
    print("AVAILABLE CHAT MODELS FOR YOU:")
    print("------------------------------------------------")
    
    try:
        for m in genai.list_models():
            # We only want models that can generate text (chat)
            if 'generateContent' in m.supported_generation_methods:
                # specifically look for gemini models
                if "gemini" in m.name:
                    print(f"- {m.name}")
    except Exception as e:
        print(f"Error listing models: {e}")