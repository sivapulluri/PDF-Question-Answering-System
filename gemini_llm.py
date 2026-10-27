
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Check your .env file."
    )

client = genai.Client(api_key=API_KEY)


def generate_answer(question, relevant_chunks):
    context = "\n\n".join(
        chunk["text"] for chunk in relevant_chunks
    )

    prompt = f"""
You are a helpful IT Helpdesk Knowledge Assistant.

Answer the user's question using only the context provided below.
If the answer is not available in the context, say:
"I could not find this information in the uploaded document."

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )

        return response.text or "No answer was generated."

    except Exception as error:
        return f"Gemini API error: {error}"