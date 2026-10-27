# AI-Powered PDF Question Answering Assistant

## 1. Project Overview

This project is an AI-powered PDF question-answering application.
Users can upload a PDF, ask questions about its content, and receive answers based on relevant passages retrieved from the document.

The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant document chunks and provide context to the Gemini language model.

## 2. Features

- Upload PDF documents.
- Extract text and retain page numbers.
- Split text into overlapping chunks.
- Generate text embeddings.
- Retrieve relevant chunks using FAISS.
- Generate answers using the Gemini API.
- Display source filename, page number, and relevant passage.
- Handle empty questions and processing errors.

## 3. Technologies Used

- Python 3.11
- Streamlit
- PyPDF
- Sentence Transformers
- FAISS
- NumPy
- Google Gemini API
- python-dotenv

## 4. Project Structure

    app.py
    pdf_loader.py
    chunking.py
    embeddings.py
    vector_store.py
    retrieval.py
    gemini_llm.py
    requirements.txt
    README.md
    .env

## 5. Installation

Create a virtual environment:

    py -3.11 -m venv .venv

Install the required packages:

    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

## 6. Configure the API Key

Create a .env file in the project folder and add:

    GEMINI_API_KEY=your_gemini_api_key_here

Replace the placeholder with your own Gemini API key. Never share your API key or upload the .env file publicly.

## 7. Run the Application

    .\.venv\Scripts\python.exe -m streamlit run app.py

Open the local URL displayed in the terminal, usually http://localhost:8501.

## 8. How to Use

1. Open the application in a browser.
2. Upload a readable PDF.
3. Click Process PDF.
4. Enter a question about the PDF.
5. Click Get Answer.
6. Review the answer and source references.

## 9. Testing

Test the application with at least eight test cases, including at least four paraphrased questions.

Suggested tests:

1. Ask a direct question answered in the PDF.
2. Ask the same question using different wording.
3. Ask about another topic in the PDF.
4. Paraphrase the third question.
5. Ask for a specific fact from a page.
6. Ask another paraphrased question.
7. Ask a question whose answer is absent from the PDF.
8. Test an empty question or an invalid/unreadable PDF.

Record actual results. Do not mark a test as passed unless you have run it.

## 10. Limitations

- Scanned PDFs may require OCR, which is not included.
- Similarity search can retrieve irrelevant chunks.
- The model may still produce unsupported answers, so testing is necessary.
- The current retrieval configuration returns up to three chunks.

## 11. Submission Checklist

Include the source code, README.md, requirements.txt, a permitted sample PDF, and test results.

Do not include the .venv folder, .env file, API keys, or cache files in the submission ZIP.
