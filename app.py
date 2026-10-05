import os
import io
import json
import uuid
import hashlib
from datetime import datetime

import requests
import streamlit as st

from pypdf import PdfReader

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "Class Notes RAG Bot"

OLLAMA_URL = "http://localhost:11434/api/generate"

OLLAMA_MODEL = "llama3.2:3b"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHROMA_PATH = "./chroma_db"

COLLECTION_NAME = "class_notes"

CHAT_FILE = "chats.json"

TOP_K = 5

# Lower distance = more similar.
# Increase slightly if your PDF retrieval is too strict.
MAX_DISTANCE = 0.65


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 16px;
        color: #777;
        margin-bottom: 25px;
    }

    .upload-success {
        padding: 12px;
        border-radius: 8px;
        background-color: #e8f5e9;
        color: #1b5e20;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .upload-error {
        padding: 12px;
        border-radius: 8px;
        background-color: #ffebee;
        color: #b71c1c;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .upload-warning {
        padding: 12px;
        border-radius: 8px;
        background-color: #fff8e1;
        color: #795548;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .bot-name {
        font-size: 23px;
        font-weight: 700;
        text-align: center;
        padding: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    return model


embedding_model = load_embedding_model()


# ============================================================
# CHROMADB
# ============================================================

@st.cache_resource
def get_chroma_collection():

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        }
    )

    return collection


collection = get_chroma_collection()


# ============================================================
# CHAT STORAGE
# ============================================================

def load_chats():

    if not os.path.exists(CHAT_FILE):
        return {}

    try:

        with open(
            CHAT_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:

        return {}


def save_chats(chats):

    with open(
        CHAT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chats,
            file,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# CREATE NEW CHAT
# ============================================================

def create_new_chat():

    chat_id = str(uuid.uuid4())

    chats = load_chats()

    chats[chat_id] = {
        "title": "New Chat",
        "created_at": datetime.now().isoformat(),
        "messages": []
    }

    save_chats(chats)

    st.session_state.current_chat_id = chat_id

    st.session_state.messages = []


# ============================================================
# LOAD EXISTING CHAT
# ============================================================

def load_chat(chat_id):

    chats = load_chats()

    if chat_id not in chats:
        return

    st.session_state.current_chat_id = chat_id

    st.session_state.messages = (
        chats[chat_id].get(
            "messages",
            []
        )
    )


# ============================================================
# SAVE CURRENT CHAT
# ============================================================

def save_current_chat():

    chat_id = st.session_state.current_chat_id

    if not chat_id:
        return

    chats = load_chats()

    if chat_id not in chats:
        return

    chats[chat_id]["messages"] = (
        st.session_state.messages
    )

    # Create title using first user message
    user_messages = [
        message
        for message in st.session_state.messages
        if message["role"] == "user"
    ]

    if user_messages:

        first_question = (
            user_messages[0]["content"]
        )

        if len(first_question) > 45:

            title = (
                first_question[:45]
                + "..."
            )

        else:

            title = first_question

        chats[chat_id]["title"] = title

    save_chats(chats)


# ============================================================
# DELETE CHAT
# ============================================================

def delete_chat(chat_id):

    chats = load_chats()

    if chat_id in chats:

        del chats[chat_id]

        save_chats(chats)

    if (
        st.session_state.current_chat_id
        == chat_id
    ):

        st.session_state.current_chat_id = None

        st.session_state.messages = []


# ============================================================
# CLEAR VECTOR DATABASE
# ============================================================

def clear_vector_database():

    try:

        existing = collection.get()

        existing_ids = existing.get(
            "ids",
            []
        )

        if existing_ids:

            collection.delete(
                ids=existing_ids
            )

    except Exception as error:

        print(
            "Could not clear ChromaDB:",
            error
        )


# ============================================================
# EXTRACT PDF TEXT
# ============================================================

def extract_pdf_text(pdf_bytes):

    pdf_file = io.BytesIO(
        pdf_bytes
    )

    reader = PdfReader(
        pdf_file
    )

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        try:

            text = page.extract_text()

            if text:

                cleaned_text = " ".join(
                    text.split()
                )

                if cleaned_text.strip():

                    pages.append(
                        {
                            "page": page_number,
                            "text": cleaned_text
                        }
                    )

        except Exception as error:

            print(
                f"Could not read page {page_number}:",
                error
            )

    return pages


# ============================================================
# CREATE TEXT CHUNKS
# ============================================================

def create_chunks(
    pages,
    chunk_size=800,
    overlap=150
):

    chunks = []

    chunk_number = 0

    step = chunk_size - overlap

    for page_data in pages:

        page_number = page_data["page"]

        text = page_data["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end]

            if chunk_text.strip():

                chunks.append(
                    {
                        "id": f"chunk_{chunk_number}",
                        "text": chunk_text,
                        "page": page_number
                    }
                )

                chunk_number += 1

            start += step

    return chunks


# ============================================================
# PROCESS PDF
# ============================================================

def process_pdf(pdf_bytes):

    # Extract text
    pages = extract_pdf_text(
        pdf_bytes
    )

    if not pages:

        return (
            False,
            "No readable text was found in the PDF."
        )

    # Create chunks
    chunks = create_chunks(
        pages
    )

    if not chunks:

        return (
            False,
            "No usable text could be extracted."
        )

    # Remove old document
    clear_vector_database()

    # Extract chunk text
    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # Generate embeddings
    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    # IDs
    ids = [
        chunk["id"]
        for chunk in chunks
    ]

    # Metadata
    metadatas = [
        {
            "page": chunk["page"],
            "source": "uploaded_class_notes"
        }
        for chunk in chunks
    ]

    # Store in ChromaDB
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    return (
        True,
        len(chunks)
    )


# ============================================================
# CHECK OLLAMA
# ============================================================

def check_ollama():

    try:

        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=5
        )

        return response.status_code == 200

    except Exception:

        return False


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_relevant_chunks(
    question
):

    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    )

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=TOP_K,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    if not results.get("documents"):

        return []

    documents = results["documents"][0]

    metadatas = results["metadatas"][0]

    distances = results["distances"][0]

    relevant_chunks = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        if distance <= MAX_DISTANCE:

            relevant_chunks.append(
                {
                    "text": document,
                    "page": metadata["page"],
                    "distance": distance
                }
            )

    return relevant_chunks


# ============================================================
# GENERATE ANSWER USING LOCAL OLLAMA
# ============================================================

def generate_answer(
    question,
    relevant_chunks
):

    # No relevant information
    if not relevant_chunks:

        return (
            "I couldn't find this information in "
            "the uploaded class notes."
        )

    # Build context
    context_parts = []

    for index, chunk in enumerate(
        relevant_chunks,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {index}
PAGE: {chunk["page"]}

{chunk["text"]}
"""
        )

    context = "\n".join(
        context_parts
    )

    # Very strict prompt
    prompt = f"""
You are a Class Notes RAG Bot.

Your job is to answer questions ONLY from the
class notes provided in the CONTEXT.

STRICT RULES:

1. Use ONLY the information in the CONTEXT.

2. Do NOT use your pretrained knowledge.

3. Do NOT use outside information.

4. Do NOT make assumptions.

5. Do NOT invent information.

6. Do NOT guess.

7. If the answer is not clearly present in the
   CONTEXT, respond exactly:

I couldn't find this information in the uploaded class notes.

8. You may combine information from multiple
   retrieved sources ONLY if that information
   exists in the CONTEXT.

9. Keep the answer simple and clear.

10. Mention the page number when possible.

QUESTION:

{question}

CONTEXT:

{context}

ANSWER:
"""

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0
                }
            },
            timeout=180
        )

        if response.status_code != 200:

            return (
                "Unable to connect to the local AI model. "
                "Please make sure Ollama is running."
            )

        data = response.json()

        answer = data.get(
            "response",
            ""
        ).strip()

        if not answer:

            return (
                "I couldn't find this information in "
                "the uploaded class notes."
            )

        return answer

    except requests.exceptions.ConnectionError:

        return (
            "❌ Ollama is not running.\n\n"
            "Please start Ollama and try again."
        )

    except requests.exceptions.Timeout:

        return (
            "⏳ The local AI model took too long "
            "to respond. Please try again."
        )

    except Exception as error:

        return (
            "❌ An error occurred while generating "
            f"the answer: {str(error)}"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # BOT NAME
    st.markdown(
        """
        <div class="bot-name">
        📚 Class Notes RAG Bot
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "Your personal AI assistant for class notes"
    )

    st.divider()

    # --------------------------------------------------------
    # NEW CHAT
    # --------------------------------------------------------

    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):

        create_new_chat()

        st.rerun()

    st.divider()

    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    st.subheader(
        "💬 Chat History"
    )

    chats = load_chats()

    if chats:

        # Most recent first
        chat_items = list(
            chats.items()
        )

        chat_items.reverse()

        for chat_id, chat_data in chat_items:

            title = chat_data.get(
                "title",
                "New Chat"
            )

            col1, col2 = st.columns(
                [5, 1]
            )

            with col1:

                if st.button(
                    title,
                    key=f"open_{chat_id}",
                    use_container_width=True
                ):

                    load_chat(
                        chat_id
                    )

                    st.rerun()

            with col2:

                if st.button(
                    "🗑️",
                    key=f"delete_{chat_id}"
                ):

                    delete_chat(
                        chat_id
                    )

                    st.rerun()

    else:

        st.caption(
            "No saved chats yet."
        )

    st.divider()

    # --------------------------------------------------------
    # PDF UPLOAD
    # --------------------------------------------------------

    st.subheader(
        "📄 Class Notes"
    )

    uploaded_file = st.file_uploader(
        "Upload your class notes",
        type=["pdf"],
        help="Upload a PDF containing your class notes."
    )

    if uploaded_file is not None:

        pdf_bytes = uploaded_file.getvalue()

        # Create unique hash for uploaded file
        file_hash = hashlib.sha256(
            pdf_bytes
        ).hexdigest()

        # Only process if it's a new PDF
        if (
            st.session_state.pdf_hash
            != file_hash
        ):

            with st.spinner(
                "Processing PDF..."
            ):

                success, result = process_pdf(
                    pdf_bytes
                )

            if success:

                st.session_state.pdf_uploaded = True

                st.session_state.pdf_name = (
                    uploaded_file.name
                )

                st.session_state.pdf_hash = (
                    file_hash
                )

                st.markdown(
                    f"""
                    <div class="upload-success">

                    ✅ <b>File successfully uploaded!</b>

                    <br><br>

                    📄 {uploaded_file.name}

                    <br>

                    🔹 {result} text chunks created.

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.session_state.pdf_uploaded = False

                st.session_state.pdf_name = None

                st.session_state.pdf_hash = None

                st.markdown(
                    f"""
                    <div class="upload-error">

                    ❌ <b>File is not added.</b>

                    <br><br>

                    {result}

                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.markdown(
                f"""
                <div class="upload-success">

                ✅ <b>File successfully uploaded!</b>

                <br><br>

                📄 {uploaded_file.name}

                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.markdown(
            """
            <div class="upload-warning">

            ❌ <b>File is not added.</b>

            <br><br>

            Please upload a PDF first.

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # CURRENT PDF STATUS
    # --------------------------------------------------------

    if st.session_state.pdf_uploaded:

        st.success(
            f"📄 {st.session_state.pdf_name}"
        )

    else:

        st.warning(
            "No class notes PDF uploaded."
        )

    st.divider()

    # --------------------------------------------------------
    # OLLAMA STATUS
    # --------------------------------------------------------

    st.subheader(
        "🤖 AI Model"
    )

    if check_ollama():

        st.success(
            f"Ollama connected\n\n{OLLAMA_MODEL}"
        )

    else:

        st.error(
            "Ollama is not running."
        )


# ============================================================
# MAIN PAGE
# ============================================================

st.markdown(
    """
    <div class="main-title">
    📚 Class Notes RAG Bot
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">

    Ask questions from your uploaded class notes.
    The bot answers using the uploaded PDF only.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# WELCOME MESSAGE
# ============================================================

if not st.session_state.messages:

    st.info(
        """
        👋 Welcome to Class Notes RAG Bot!

        **How to use:**

        1. Upload your class notes PDF from the sidebar.
        2. Wait until the file shows "successfully uploaded".
        3. Ask a question below.
        4. The bot will search your notes.
        5. If the answer isn't in the notes, it will tell you.
        """
    )


# ============================================================
# DISPLAY EXISTING CHAT
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question from your class notes..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # CHECK PDF
    # --------------------------------------------------------

    if not st.session_state.pdf_uploaded:

        st.warning(
            "📄 Please upload your class notes PDF first."
        )

        st.stop()

    # --------------------------------------------------------
    # CHECK OLLAMA
    # --------------------------------------------------------

    if not check_ollama():

        st.error(
            "🤖 Ollama is not running. "
            "Please start Ollama and try again."
        )

        st.stop()

    # --------------------------------------------------------
    # CREATE CHAT IF NEEDED
    # --------------------------------------------------------

    if not st.session_state.current_chat_id:

        create_new_chat()

    # --------------------------------------------------------
    # ADD USER MESSAGE
    # --------------------------------------------------------

    user_message = {
        "role": "user",
        "content": question
    }

    st.session_state.messages.append(
        user_message
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )

    # --------------------------------------------------------
    # SEARCH PDF
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "🔎 Searching your class notes..."
        ):

            relevant_chunks = (
                retrieve_relevant_chunks(
                    question
                )
            )

        # ----------------------------------------------------
        # GENERATE ANSWER
        # ----------------------------------------------------

        with st.spinner(
            "🤖 Generating answer from your notes..."
        ):

            answer = generate_answer(
                question,
                relevant_chunks
            )

        st.markdown(
            answer
        )

        # ----------------------------------------------------
        # SHOW SOURCES
        # ----------------------------------------------------

        if relevant_chunks:

            pages = sorted(
                set(
                    chunk["page"]
                    for chunk in relevant_chunks
                )
            )

            page_text = ", ".join(
                str(page)
                for page in pages
            )

            st.caption(
                f"📑 Retrieved from PDF page(s): {page_text}"
            )

    # --------------------------------------------------------
    # SAVE ASSISTANT MESSAGE
    # --------------------------------------------------------

    assistant_message = {
        "role": "assistant",
        "content": answer
    }

    st.session_state.messages.append(
        assistant_message
    )

    # --------------------------------------------------------
    # SAVE CHAT
    # --------------------------------------------------------

    save_current_chat()