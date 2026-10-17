# Enterprise Document RAG System

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their contents using natural language.

The system retrieves relevant information from uploaded documents and uses a Large Language Model (LLM) to generate context-aware answers with source references.

## 1. Project Overview

Searching large PDF documents manually can be time-consuming. Traditional keyword-based search may also fail to identify relevant information when the user's query uses different wording from the document.

The Enterprise Document RAG System addresses this problem by combining semantic search with Large Language Model-based answer generation.

Users can upload PDF documents, process their contents, and ask questions. The application retrieves relevant text passages from the documents and uses them as context for generating answers.

### Key Features

- **PDF Upload:** Upload multiple PDF documents.
- **Document Processing:** Extract text and divide it into manageable chunks.
- **Semantic Search:** Find relevant passages using vector embeddings.
- **Vector Storage:** Store document embeddings in ChromaDB.
- **AI-Powered Question Answering:** Generate answers using Ollama Cloud.
- **Source References:** Display the source documents and page numbers associated with retrieved information.
- **Context-Aware Responses:** Instruct the LLM to answer using retrieved context and acknowledge when the information is unavailable.
- **Performance Evaluation:** Measure retrieval time, response time, and other system metrics.

## 2. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | User interface |
| LangChain | RAG pipeline orchestration |
| Sentence Transformers | Generate text embeddings |
| ChromaDB | Store and retrieve document embeddings |
| Ollama Cloud | Generate answers using an LLM |
| PyPDF | Extract text from PDF documents |
| Git and GitHub | Version control and project hosting |

### Embedding Model

`sentence-transformers/all-MiniLM-L6-v2`

This model converts text into numerical vector representations that allow semantically similar passages to be retrieved.

### Language Model

`gpt-oss:120b-cloud`

The application uses Ollama Cloud for answer generation. Model availability and access requirements depend on the current Ollama service and account.

## 3. System Architecture

The application follows a Retrieval-Augmented Generation architecture.

### Document Indexing Workflow

1. The user uploads one or more PDF files.
2. PyPDF extracts text from the documents.
3. LangChain splits the extracted text into smaller chunks.
4. Sentence Transformers generates embeddings for the chunks.
5. ChromaDB stores the embeddings and associated document metadata.

### Question-Answering Workflow

1. The user enters a question through the Streamlit interface.
2. The system searches ChromaDB for relevant document chunks.
3. The top four retrieved chunks are selected as context.
4. The question and retrieved context are sent to the Ollama Cloud model.
5. The model generates a context-based answer.
6. The application displays the answer and available source references.

### Architecture Diagram

```text
              USER
                |
                v
       +------------------+
       |  Streamlit UI    |
       +------------------+
                |
        +-------+--------+
        |                |
        v                v
  Upload PDF         Ask Question
        |                |
        v                v
  +-------------+  +----------------+
  | PyPDF       |  | ChromaDB       |
  | Text Extract|  | Semantic Search|
  +-------------+  +----------------+
        |                |
        v                v
  +-------------+  Retrieved Chunks
  | Text        |        |
  | Chunking    |        |
  +-------------+        |
        |                |
        v                v
  +-------------+  +----------------+
  | Sentence    |  | Ollama Cloud    |
  | Transformers|  | LLM             |
  +-------------+  +----------------+
        |                |
        v                v
  +-------------+  Generated Answer
  | ChromaDB    |        |
  | Vector Store|        v
  +-------------+  +----------------+
                   | Answer and     |
                   | Source Details |
                   +----------------+
```

Document indexing takes place when documents are processed. Question answering uses the existing vector store.

## 4. Project Structure

```text
enterprise-document-rag/
│
├── app.py
├── rag.py
├── requirements.txt
├── .gitignore
├── README.md
│
└── data/
    └── chroma_db/
        └── Generated vector database files
```

### File Descriptions

| File or Directory | Description |
|---|---|
| `app.py` | Implements the Streamlit user interface |
| `rag.py` | Implements PDF processing, embedding generation, retrieval, and answer generation |
| `requirements.txt` | Lists the Python dependencies |
| `.gitignore` | Specifies files and directories excluded from Git |
| `data/chroma_db/` | Stores the generated ChromaDB database |

The vector database is generated when documents are processed and does not need to be committed to GitHub.

## 5. Prerequisites

Install the following before running the project:

- Python 3.10 or a compatible version supported by the installed dependencies
- pip
- Git
- An Ollama account with access to the required cloud model
- An internet connection for downloading dependencies and using Ollama Cloud

Check your Python and pip installations:

```bash
python3 --version
python3 -m pip --version
```

Install Ollama by following the instructions on the official website:

https://ollama.com/download

## 6. Installation

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Navigate to the project directory:

```bash
cd enterprise-document-rag
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with your actual GitHub repository URL.

### Step 2: Create a Virtual Environment

```bash
python3 -m venv .venv
```

Activate the environment on Linux:

```bash
source .venv/bin/activate
```

On Windows, use:

```powershell
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Ollama Cloud

Sign in to Ollama using its CLI:

```bash
ollama signin
```

Pull the required cloud model:

```bash
ollama pull gpt-oss:120b-cloud
```

Follow Ollama's current documentation if its authentication or cloud-model commands change.

Ollama Cloud is used for LLM inference. The embedding model and ChromaDB vector store are configured to run locally.

### Step 5: Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal. Streamlit commonly uses:

```text
http://localhost:8501
```

## 7. How to Use the Application

1. Launch the application.
2. Upload one or more PDF documents.
3. Click the document-processing button.
4. Wait for the documents to be processed and indexed.
5. Enter a question about the uploaded documents.
6. Submit the question.
7. Read the generated answer.
8. Expand the source details to inspect the retrieved document references.

### Example

Suppose you upload an employee handbook containing leave policies.

**Question:**

```text
How many days of annual leave are provided to employees?
```

**Expected behaviour:**

The system searches the uploaded handbook, retrieves relevant passages, and generates an answer based on the retrieved content.

The actual answer depends on the contents of the uploaded PDF.

If the documents do not contain the requested information, the system is instructed to acknowledge the limitation instead of inventing an answer. However, this behaviour is not guaranteed and should be tested.

## 8. Model Implementation

### Text Embeddings

The Sentence Transformers model converts document chunks into numerical vectors.

These vectors represent semantic information and allow the system to identify passages related to a question even when the wording differs.

### Text Chunking

The current implementation uses the following initial configuration:

| Parameter | Value |
|---|---:|
| Chunk size | 1,000 characters |
| Chunk overlap | 150 characters |
| Number of retrieved chunks | 4 |
| Embedding model | `all-MiniLM-L6-v2` |

Chunk overlap preserves some context between consecutive chunks.

These values are configuration choices, not experimentally proven optimal settings.

### Vector Database

ChromaDB stores the document embeddings alongside metadata such as source filenames and page numbers.

When a question is submitted, the system searches the vector database and retrieves the most relevant chunks.

### Answer Generation

The retrieved chunks and the user's question are provided to the Ollama Cloud model.

The prompt instructs the model to use the supplied context, avoid unsupported claims, and acknowledge missing information.

The application then displays the generated response and source references.

## 9. Evaluation Metrics

The system can be evaluated using both performance measurements and answer-quality metrics.

| Metric | Purpose |
|---|---|
| Number of documents indexed | Measures the number of processed PDF files |
| Number of chunks | Measures the number of stored document segments |
| Average retrieval time | Measures the time required to retrieve relevant chunks |
| Average response time | Measures the time required to generate an answer |
| Recall@K | Measures whether relevant passages appear among the top K retrieved results |
| Mean Reciprocal Rank (MRR) | Measures the ranking of the first relevant result |
| Answer correctness | Measures agreement between generated answers and reference answers |
| Faithfulness | Measures whether generated claims are supported by retrieved context |
| Answerability F1-score | Evaluates a separately defined task that classifies questions as answerable or unanswerable |

### Evaluation Procedure

1. Prepare a test dataset of questions based on the uploaded documents.
2. Record the expected answers and relevant source passages.
3. Include questions whose answers are absent from the documents.
4. Run the application against the test questions.
5. Record retrieval and response times.
6. Evaluate answer correctness and faithfulness using a defined rubric.
7. Calculate retrieval and classification metrics where applicable.
8. Summarize the measured results in tables or graphs.

**Note:** Accuracy, precision, recall, and F1-score require a clearly defined classification or labelled evaluation task. They should not be treated as automatic measures of free-form answer quality.

Actual evaluation results should be added after testing the implemented system.

## 10. Limitations

- **PDF format:** The current implementation primarily supports text-based PDFs.
- **Scanned documents:** Image-only PDFs may require OCR before their contents can be searched.
- **Retrieval quality:** Relevant information may not appear in the retrieved chunks.
- **LLM reliability:** Generated answers may contain incorrect or unsupported statements.
- **Cloud dependency:** Answer generation requires access to the Ollama Cloud service.
- **Privacy:** Retrieved document text is sent to the cloud model for inference. Do not upload confidential documents unless their use with the service is permitted.
- **Index replacement:** The current indexing implementation replaces the existing vector database when documents are reprocessed. It is not designed for incremental document updates.
- **Evaluation:** Reliable quality measurements require a representative test dataset and consistent evaluation criteria.

## 11. Future Enhancements

Potential improvements include:

- Support for scanned PDFs through OCR.
- Support for DOCX and TXT documents.
- Hybrid search combining keyword-based and semantic retrieval.
- Improved source citations and passage-level references.
- Document-level access control.
- Incremental indexing and document deletion.
- Downloadable chat history.
- Automated evaluation using a larger benchmark dataset.
- Comparison of different embedding models and chunking strategies.
- Enhanced monitoring and performance visualizations.

## 12. Privacy and Security

This project is intended as an academic prototype.

Before uploading documents, ensure that you have permission to process their contents.

The vector database is stored locally, but retrieved passages and questions are sent to Ollama Cloud for answer generation. Local vector storage does not mean the entire workflow is local.

Do not commit private documents, API credentials, authentication files, or generated private datasets to a public GitHub repository.

## 13. GitHub License

This project can be distributed under the MIT License.

The MIT License permits use, modification, distribution, and redistribution, subject to its terms and preservation of the required copyright and license notices.

To apply it, create a file named `LICENSE` in the project root and include the standard MIT License text with the appropriate copyright holder and year.

Review the licenses and terms applicable to the project's dependencies, embedding model, LLM, and cloud service separately.

## 14. References

1. P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," *Advances in Neural Information Processing Systems*, vol. 33, pp. 9459–9474, 2020. Available: https://arxiv.org/abs/2005.11401

2. N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," in *Proceedings of EMNLP-IJCNLP 2019*, pp. 3982–3992, 2019. Available: https://aclanthology.org/D19-1410/

3. LangChain Documentation. Available: https://docs.langchain.com/

4. Sentence Transformers Documentation. Available: https://sbert.net/docs/

5. Chroma Documentation. Available: https://docs.trychroma.com/

6. Streamlit Documentation. Available: https://docs.streamlit.io/

7. Ollama Documentation. Available: https://ollama.com/

## 15. Author

**Project:** Enterprise Document RAG System  
**Domain:** Artificial Intelligence and Natural Language Processing  
**Application Type:** Document Question Answering System  
**Interface:** Streamlit Web Application  
**Core Approach:** Retrieval-Augmented Generation (RAG)

---

*Developed as an academic project to explore semantic search, vector databases, embeddings, and Large Language Model-based question answering.*
