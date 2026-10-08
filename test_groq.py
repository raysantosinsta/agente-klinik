from dotenv import load_dotenv
load_dotenv(".env")
import os
from groq import Groq
client = Groq()
try:
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": "oi"}],
    )
    print("SUCCESS:", completion.choices[0].message.content)
except Exception as e:
    print("ERROR:", str(e))
