import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

# Global cached client instance
_cached_client = None


def get_gemini_api_key():
    """
    Retrieve Gemini API key from environment variables or Streamlit secrets.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            if "GEMINI_API_KEY" in st.secrets:
                api_key = st.secrets["GEMINI_API_KEY"]
            elif "gemini" in st.secrets and "api_key" in st.secrets["gemini"]:
                api_key = st.secrets["gemini"]["api_key"]
        except Exception:
            pass
    return api_key


def get_gemini_client():
    """
    Get or create a cached Google GenAI Client.
    Initializes lazily to prevent startup crashes when keys are missing.
    """
    global _cached_client
    if _cached_client is not None:
        return _cached_client

    api_key = get_gemini_api_key()
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Please set GEMINI_API_KEY as an "
            "environment variable in Render or in your .env / Streamlit secrets."
        )

    _cached_client = genai.Client(api_key=api_key)
    return _cached_client


def generate_answer(question=None, relevant_chunks=None, query=None, context=None):
    """
    Generate an answer grounded strictly in the provided relevant chunks using Gemini.
    Supports both (question, relevant_chunks) and (query, context) argument patterns.
    """
    if question is None and query is not None:
        question = query

    if context is None:
        context = "\n\n".join(
            chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            for chunk in (relevant_chunks or [])
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
        client = get_gemini_client()
        # Primary model is gemini-3.5-flash-lite, fallback to 3.8-flash or configured model
        preferred_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        fallback_models = [preferred_model, "gemini-3.8-flash", "gemini-flash-latest"]

        last_error = None
        for model_name in fallback_models:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as model_err:
                last_error = model_err
                continue

        if last_error:
            return f"Gemini API error: {last_error}"
        return "No answer was generated."

    except Exception as error:
        return f"Gemini API error: {error}"