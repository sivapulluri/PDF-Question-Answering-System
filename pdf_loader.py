from pypdf import PdfReader
from io import BytesIO


def load_pdf(uploaded_file):
    """
    Extract text from each PDF page.
    Preserve filename and page number.
    """

    documents = []

    try:
        pdf_bytes = BytesIO(uploaded_file.getvalue())
        reader = PdfReader(pdf_bytes)

        if len(reader.pages) == 0:
            raise ValueError("The uploaded PDF contains no pages.")

        for page_number, page in enumerate(reader.pages, start=1):

            page_text = page.extract_text()

            if page_text and page_text.strip():

                documents.append({
                    "text": page_text.strip(),
                    "source": uploaded_file.name,
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
        raise ValueError(
            f"Unable to process the PDF: {e}"
        ) from e
