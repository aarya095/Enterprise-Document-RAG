import streamlit as st

from rag import (
    index_documents,
    get_vectorstore,
    answer_question,
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Enterprise Document RAG",
    page_icon="📚",
    layout="wide",
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed" not in st.session_state:
    st.session_state.indexed = False


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📚 Enterprise Document Assistant")

st.write(
    "Upload PDF documents and ask questions about their contents. "
    "Answers are generated using a locally running language model."
)

st.divider()


# --------------------------------------------------
# SIDEBAR: DOCUMENT MANAGEMENT
# --------------------------------------------------

with st.sidebar:

    st.header("Document Management")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload all PDFs you want to search together.",
    )

    st.caption(
        "Indexing replaces the existing collection. "
        "Upload all required PDFs before indexing."
    )

    if st.button(
        "Process Documents",
        type="primary",
        use_container_width=True,
    ):

        if not uploaded_files:

            st.warning("Please upload at least one PDF.")

        else:

            try:
                with st.spinner(
                    "Extracting text, generating embeddings "
                    "and building the search index..."
                ):

                    vectorstore, chunk_count = index_documents(
                        uploaded_files
                    )

                st.session_state.indexed = True
                st.session_state.messages = []

                st.success(
                    f"Successfully indexed {len(uploaded_files)} "
                    f"PDF(s) into {chunk_count} text chunks."
                )

            except Exception as error:

                st.error(
                    f"Document processing failed: {error}"
                )

    st.divider()

    st.subheader("System Status")

    # Detect an existing index, including after a page refresh.
    vectorstore = get_vectorstore()

    if vectorstore is not None:

        st.success("Document index available")

    else:

        st.info("No documents indexed yet")

    st.caption("Embedding model: all-MiniLM-L6-v2")
    st.caption("LLM: gpt-oss:120b-cloud")
    st.caption("Vector database: ChromaDB")


# --------------------------------------------------
# MAIN CHAT INTERFACE
# --------------------------------------------------

st.subheader("Ask Your Documents")

if vectorstore is None:

    st.info(
        "Upload your PDFs in the sidebar and click "
        "'Process Documents' to get started."
    )

else:

    # Display previous messages.
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources beneath assistant responses.
            if message["role"] == "assistant":

                sources = message.get("sources", [])

                if sources:

                    with st.expander("View source documents"):

                        for source in sources:

                            st.markdown(
                                f"**{source['filename']}** "
                                f"— Page {source['page']}"
                            )

                            st.caption(
                                source["content"][:500]
                            )

                            st.divider()

    # Accept a new question.
    question = st.chat_input(
        "Ask a question about your documents..."
    )

    if question:

        # Display the user's question.
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        # Retrieve context and generate an answer.
        with st.chat_message("assistant"):

            try:

                with st.spinner(
                    "Searching documents and generating an answer..."
                ):

                    answer, sources = answer_question(
                        question,
                        vectorstore,
                    )

                st.markdown(answer)

                if sources:

                    with st.expander("View source documents"):

                        for source in sources:

                            st.markdown(
                                f"**{source['filename']}** "
                                f"— Page {source['page']}"
                            )

                            st.caption(
                                source["content"][:500]
                            )

                            st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as error:

                error_message = (
                    "Unable to generate an answer. "
                    "Check that Ollama is running and the "
                    f"configured model is installed. Details: {error}"
                )

                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                })