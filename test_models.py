from google import genai
import os

API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(
    

API_KEY = os.getenv("GEMINI_API_KEY")
)

for model in client.models.list():
    print(model.name)