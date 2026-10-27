import streamlit as st
import pandas as pd
import json
import hmac

from database import init_db, save_chat, get_all_chats

from pdf_loader import load_pdf
from chunking import create_chunks
from embeddings import create_embeddings
from vector_store import create_vector_store
from retrieval import retrieve_relevant_chunks
from gemini_llm import generate_answer


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="PDF Knowledge Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
init_db()


# ==========================================
# ADMIN AUTHENTICATION
# ==========================================

def check_admin_password(password):
    try:
        correct_password = st.secrets["admin"]["password"]
        return hmac.compare_digest(password, correct_password)
    except (KeyError, FileNotFoundError):
        return False


# ==========================================
# DARK THEME CSS
# ==========================================

st.markdown(
    """
    <style>
    .stApp,
    [data-testid="stMain"] {
        background-color: #0B0F19;
        color: #F9FAFB;
    }

    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {
        background-color: #111827 !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #F9FAFB !important;
    }

    h1, h2, h3, h4, p, label {
        color: #F9FAFB;
    }

    [data-testid="stFileUploader"] {
        background-color: #1F2937;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 10px;
    }

    [data-testid="stFileUploader"] section {
        background-color: #1F2937 !important;
        border-color: #4B5563 !important;
    }

    [data-testid="stFileUploader"] button {
        background-color: #374151 !important;
        color: white !important;
    }

    .stButton > button {
        background-color: #2563EB;
        color: white !important;
        border: none;
        border-radius: 9px;
        padding: 8px 16px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #1D4ED8;
        color: white !important;
    }

    [data-testid="stChatMessage"] {
        background-color: #151C2B;
        border: 1px solid #263244;
        border-radius: 14px;
        padding: 16px;
    }

    [data-testid="stChatInput"] {
        background-color: #111827;
        border: 1px solid #374151;
        border-radius: 12px;
    }

    [data-testid="stChatInput"] textarea {
        background-color: #111827 !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    .welcome-card {
        background-color: #111827;
        border: 1px solid #263244;
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 20px;
    }

    .info-card {
        background-color: #1F2937;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
    }

    .source-box {
        background-color: #1F2937;
        border-left: 4px solid #2563EB;
        border-radius: 8px;
        padding: 12px;
        margin-top: 10px;
    }

    hr {
        border-color: #374151;
    }

    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    a {
        color: #60A5FA !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# SESSION STATE
# ==========================================

defaults = {
    "messages": [],
    "processed": False,
    "vector_store": None,
    "embedding_model": None,
    "chunks": [],
    "uploaded_filename": None,
    "username": "",
    "admin_authenticated": False,
    "page": "User Chat"
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

with st.sidebar:

    st.markdown("# 📚 PDF Assistant")

    st.markdown(
        """
        <p style="color:#9CA3AF;">
        Ask questions from your PDF documents.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page_options = ["User Chat", "Admin Dashboard"]

    selected_page = st.radio(
        "Navigation",
        page_options,
        index=page_options.index(st.session_state.page)
        if st.session_state.page in page_options else 0
    )

    st.session_state.page = selected_page

    st.divider()


# ==========================================
# ADMIN LOGIN
# ==========================================

if st.session_state.page == "Admin Dashboard":

    st.markdown(
        """
        <div class="welcome-card">
            <h1>🔐 Admin Dashboard</h1>
            <p style="color:#9CA3AF;">
                Authorized access to chat monitoring and activity logs.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not st.session_state.admin_authenticated:

        st.subheader("Admin Login")

        with st.form("admin_login_form"):

            admin_password = st.text_input(
                "Admin Password",
                type="password"
            )

            login_button = st.form_submit_button(
                "Login as Admin",
                use_container_width=True
            )

        if login_button:

            if check_admin_password(admin_password):
                st.session_state.admin_authenticated = True
                st.success("Admin login successful!")
                st.rerun()

            else:
                st.error(
                    "Invalid admin password. "
                    "Check your Streamlit secrets configuration."
                )

        st.info(
            "Configure the admin password in "
            "`.streamlit/secrets.toml` before logging in."
        )

    else:

        col1, col2 = st.columns([5, 1])

        with col1:
            st.success("Authenticated as Admin")

        with col2:
            if st.button("Logout"):
                st.session_state.admin_authenticated = False
                st.session_state.page = "User Chat"
                st.rerun()

        # Retrieve logs
        records = get_all_chats()

        columns = [
            "Username",
            "Question",
            "Answer",
            "PDF Name",
            "Sources",
            "Status",
            "Timestamp"
        ]

        df = pd.DataFrame(records, columns=columns)

        st.divider()
        st.subheader("📊 Activity Overview")

        total_questions = len(df)

        if total_questions:
            successful = int((df["Status"] == "Success").sum())
            failed = int((df["Status"] == "Failed").sum())
            unique_users = int(df["Username"].nunique())
        else:
            successful = 0
            failed = 0
            unique_users = 0

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Questions", total_questions)
        c2.metric("Successful", successful)
        c3.metric("Failed", failed)
        c4.metric("Unique Usernames", unique_users)

        st.divider()
        st.subheader("🔎 Search and Filter")

        search_text = st.text_input(
            "Search questions, answers, usernames or PDF names"
        )

        if total_questions:

            pdf_options = ["All PDFs"] + sorted(
                df["PDF Name"].dropna().unique().tolist()
            )

            selected_pdf = st.selectbox(
                "Filter by PDF",
                pdf_options
            )

            status_options = ["All Statuses"] + sorted(
                df["Status"].dropna().unique().tolist()
            )

            selected_status = st.selectbox(
                "Filter by Status",
                status_options
            )

            filtered_df = df.copy()

            if selected_pdf != "All PDFs":
                filtered_df = filtered_df[
                    filtered_df["PDF Name"] == selected_pdf
                ]

            if selected_status != "All Statuses":
                filtered_df = filtered_df[
                    filtered_df["Status"] == selected_status
                ]

            if search_text.strip():

                search_columns = [
                    "Username",
                    "Question",
                    "Answer",
                    "PDF Name"
                ]

                mask = filtered_df[search_columns].fillna("").apply(
                    lambda column: column.astype(str).str.contains(
                        search_text,
                        case=False,
                        regex=False
                    )
                ).any(axis=1)

                filtered_df = filtered_df[mask]

            st.subheader("📋 User Chat Logs")

            st.caption(
                f"Showing {len(filtered_df)} of {total_questions} records."
            )

            st.dataframe(
                filtered_df,
                use_container_width=True,
                hide_index=True
            )

            csv_data = filtered_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇️ Download Filtered Logs as CSV",
                data=csv_data,
                file_name="chat_logs.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.divider()
            st.subheader("💬 Individual Conversation Details")

            for index, row in filtered_df.iterrows():

                label = (
                    f"{row['Timestamp']} | "
                    f"{row['Username']} | "
                    f"{str(row['Question'])[:80]}"
                )

                with st.expander(label):

                    st.markdown("**Username**")
                    st.write(row["Username"])

                    st.markdown("**Question**")
                    st.write(row["Question"])

                    st.markdown("**Generated Answer**")
                    st.write(row["Answer"] or "No answer recorded.")

                    st.markdown("**PDF Name**")
                    st.write(row["PDF Name"] or "Unknown")

                    st.markdown("**Sources**")
                    st.write(row["Sources"] or "No sources recorded.")

                    st.markdown("**Status**")
                    st.write(row["Status"])

                    st.markdown("**Timestamp**")
                    st.write(row["Timestamp"])

        else:
            st.info(
                "No chat logs available yet. "
                "User questions will appear here after they are submitted."
            )

    # Stop user-chat rendering on the admin page.
    st.stop()


# ==========================================
# USER IDENTITY
# ==========================================

with st.sidebar:

    st.subheader("👤 User")

    username_input = st.text_input(
        "Enter your username",
        value=st.session_state.username,
        placeholder="e.g. user01"
    )

    if username_input.strip():
        st.session_state.username = username_input.strip()

    st.caption(
        "Enter a username before asking questions. "
        "Use the same username to identify your activity."
    )


# ==========================================
# USER CHAT SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("### 📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )

    if st.button("⚙️ Process PDF", use_container_width=True):

        if uploaded_file is None:
            st.warning("Please upload a PDF file first.")

        else:
            try:

                with st.spinner("Reading PDF..."):
                    documents = load_pdf(uploaded_file)

                with st.spinner("Splitting into chunks..."):
                    chunks = create_chunks(documents)

                if not chunks:
                    st.error("No text chunks were created.")

                else:

                    texts = [chunk["text"] for chunk in chunks]

                    with st.spinner("Creating embeddings..."):
                        embeddings, embedding_model = create_embeddings(texts)

                    with st.spinner("Building vector store..."):
                        vector_store = create_vector_store(embeddings)

                    st.session_state.chunks = chunks
                    st.session_state.embedding_model = embedding_model
                    st.session_state.vector_store = vector_store
                    st.session_state.processed = True
                    st.session_state.uploaded_filename = uploaded_file.name
                    st.session_state.messages = []

                    st.success("PDF processed successfully!")
                    st.rerun()

            except Exception as error:
                st.error(f"Error processing PDF: {error}")

    st.divider()

    st.markdown("### 📊 Document Status")

    if st.session_state.processed:

        st.success("✅ Document Ready")

        st.markdown(
            f"""
            <div class="info-card">
                <p><b>File:</b> {st.session_state.uploaded_filename}</p>
                <hr>
                <p><b>Total Chunks:</b> {len(st.session_state.chunks)}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.info("Upload and process a PDF to begin.")

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ==========================================
# MAIN HEADER
# ==========================================

st.markdown(
    """
    <div class="welcome-card">
        <h1>🤖 PDF Knowledge Assistant</h1>
        <p style="color:#9CA3AF; font-size:17px;">
            Your AI-powered assistant for PDF question answering.
        </p>
        <p style="color:#D1D5DB;">
            Upload a document, process it, and ask questions
            based on its content.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================
# INITIAL SCREEN
# ==========================================

if not st.session_state.processed:

    st.info(
        "Upload a PDF from the left sidebar and click "
        "**Process PDF** to start."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="info-card">
                <h3>📄 Upload</h3>
                <p>Upload your PDF document.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="info-card">
                <h3>🔍 Retrieve</h3>
                <p>Find relevant information in your document.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="info-card">
                <h3>💬 Ask</h3>
                <p>Ask questions and get answers with sources.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.stop()


# ==========================================
# EXAMPLE QUESTIONS
# ==========================================

st.markdown("### 💡 Example Questions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("What is this document about?", use_container_width=True):
        st.session_state.pending_question = "What is this document about?"

with col2:
    if st.button("Summarize the main points.", use_container_width=True):
        st.session_state.pending_question = "Summarize the main points."

with col3:
    if st.button("Explain the important information.", use_container_width=True):
        st.session_state.pending_question = (
            "Explain the important information in this PDF."
        )


# ==========================================
# CHAT HISTORY
# ==========================================

st.markdown("### 💬 Conversation")

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("📚 View Sources"):

                for source in message["sources"]:
                    st.markdown(
                        f'<div class="source-box">{source}</div>',
                        unsafe_allow_html=True
                    )


# ==========================================
# CHAT INPUT
# ==========================================

user_question = st.chat_input("Ask a question about your PDF...")

if "pending_question" in st.session_state:
    user_question = st.session_state.pop("pending_question")


# ==========================================
# ANSWER + DATABASE LOGGING
# ==========================================

if user_question:

    username = st.session_state.username.strip()

    if not username:
        st.warning("Please enter your username in the sidebar first.")
        st.stop()

    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):

        answer = ""
        sources = []
        status = "Failed"

        try:

            with st.spinner("Searching the document..."):

                relevant_chunks = retrieve_relevant_chunks(
                    user_question,
                    st.session_state.embedding_model,
                    st.session_state.vector_store,
                    st.session_state.chunks
                )

            if not relevant_chunks:

                answer = (
                    "I couldn't find relevant information in "
                    "the uploaded PDF to answer this question."
                )

            else:

                with st.spinner("Generating answer..."):

                    answer = generate_answer(
                        user_question,
                        relevant_chunks
                    )

                seen_sources = set()

                for chunk in relevant_chunks:

                    if not isinstance(chunk, dict):
                        continue

                    filename = chunk.get(
                        "source",
                        st.session_state.uploaded_filename or "Uploaded PDF"
                    )

                    page = chunk.get("page")

                    if page is None:
                        source_text = f"**{filename}** — Page number unavailable"
                    else:
                        source_text = f"**{filename}** — Page {page}"

                    if source_text not in seen_sources:
                        sources.append(source_text)
                        seen_sources.add(source_text)

            status = "Success"

            st.markdown(answer)

            if sources:

                with st.expander("📚 View Sources", expanded=True):

                    for source in sources:
                        st.markdown(
                            f'<div class="source-box">{source}</div>',
                            unsafe_allow_html=True
                        )

            else:
                st.caption(
                    "No source page information was returned for this answer."
                )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        except Exception as error:

            answer = f"Sorry, something went wrong: {error}"
            status = "Failed"

            st.error(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": []
            })

        finally:

            try:
                save_chat(
                    username=username,
                    question=user_question,
                    answer=answer,
                    pdf_name=st.session_state.uploaded_filename or "Not uploaded",
                    sources=json.dumps(sources, ensure_ascii=False),
                    status=status
                )

            except Exception as db_error:
                st.error(f"Could not save chat history: {db_error}")