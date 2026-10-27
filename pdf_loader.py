import os
from pypdf import PdfReader
from io import BytesIO


def load_pdf(uploaded_file):
    """
    Extract text from each PDF page.
    Preserve filename and page number.
    Supports Streamlit UploadedFile, file-like objects, or file paths.
    """
    documents = []

    try:
        if hasattr(uploaded_file, "getvalue"):
            pdf_bytes = BytesIO(uploaded_file.getvalue())
            filename = getattr(uploaded_file, "name", "Uploaded PDF")
        elif isinstance(uploaded_file, (str, bytes, os.PathLike)):
            if isinstance(uploaded_file, (str, os.PathLike)):
                with open(uploaded_file, "rb") as f:
                    pdf_bytes = BytesIO(f.read())
                filename = os.path.basename(str(uploaded_file))
            else:
                pdf_bytes = BytesIO(uploaded_file)
                filename = "Uploaded PDF"
        elif hasattr(uploaded_file, "read"):
            pdf_bytes = BytesIO(uploaded_file.read())
            filename = getattr(uploaded_file, "name", "Uploaded PDF")
        else:
            raise ValueError("Unsupported file input format.")

        reader = PdfReader(pdf_bytes)

        if len(reader.pages) == 0:
            raise ValueError("The uploaded PDF contains no pages.")

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                documents.append({
                    "text": page_text.strip(),
                    "source": filename,
                    "page": page_number
                })

        if not documents:
            raise ValueError(
                "No readable text found in this PDF. "
                "The PDF may be scanned or image-based."
            )

        return documents

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Unable to process the PDF: {e}") from e


def load_pdfs(folder_path):
    """
    Load all PDF files in a given directory folder.
    """
    all_documents = []
    if not os.path.exists(folder_path):
        return all_documents

    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".pdf"):
            full_path = os.path.join(folder_path, fname)
            try:
                docs = load_pdf(full_path)
                all_documents.extend(docs)
            except Exception as e:
                print(f"Skipping {fname}: {e}")

    return all_documents
