from google import genai
import os

API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)

models_to_test = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite"
]

for model in models_to_test:
    try:
        response = client.models.generate_content(
            model=model,
            contents="Say hello."
        )
        print(f"✅ {model} works")
    except Exception as e:
        print(f"❌ {model} failed")
        print(e)