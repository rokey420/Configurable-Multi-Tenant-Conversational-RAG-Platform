# Configurable Multi-Tenant Conversational RAG Platform

A production-style **multi-tenant Retrieval-Augmented Generation (RAG)** platform where **organizations (admins)** can create **role-specific topics/agents**, upload proprietary documents, and give **employees/users** access to chat with **citation-grounded** answers.

The system supports:
- **Per-topic document isolation** using **Pinecone namespaces**
- **Per-user chat history** stored in **PostgreSQL**
- **Hybrid conversational behavior**:
  - Small talk 
  - Memory questions from chat history (“what was my first question?”)
  - RAG answers grounded in uploaded docs  (with citations)
  - Safe fallback when documents don’t contain the answer 

---

## Key Features

###  Multi-Tenant Topics (Role-Specific Agents)
- Admin creates **topics** like “HR Policy”, “Engineering Wiki”, “Onboarding”, etc.
- Each topic has a **behavior prompt** that defines role, tone, and rules.
- **Employees only see topics they are added to**.

###  Secure Admin + User Modes (Product-Style)
- **Admin mode**:
  - One-time bootstrap “Create First Admin”
  - Admin login via token (`X-Admin-Token`)
  - Create topics, upload documents, add employees to topics
- **User mode**:
  - Login via username
  - Select topic
  - Chat with RAG + memory + fallback

###  Citation-Grounded Answers (RAG)
- Documents are chunked and embedded with **MiniLM**.
- Retrieval via **Pinecone vector search**.
- Results reranked with **Cross-Encoder (ms-marco-MiniLM-L-6-v2)**.
- Final answer includes **citations** (source + page when available).

###  Conversation Memory (PostgreSQL)
- Stores per-user (and per-topic) chat history.
- Supports memory-only questions like:
  - “What was my first question?”
  - “Summarize our conversation”
  - “What did I say last time?”

### Safe Fallback (When Docs Don’t Contain Answer)
When context doesn’t contain the answer:
> “I don’t know this based on the provided documents. But here is what I know about it: …”

This makes the chatbot useful even when retrieval is empty, while remaining honest.

---

## Tech Stack

**Backend**
- FastAPI
- PostgreSQL (chat history, users, topics, membership, admin sessions)
- Pinecone (vector DB + namespaces per topic)
- LangChain + Gemini (response generation)
- HuggingFace MiniLM embeddings
- Cross-Encoder reranker

**Frontend**
- Streamlit (Admin/User UI)

**Evaluation**
- `eval.py` to compute:
  - Recall@5 / Recall@10
  - MRR
  - Retrieval latency & response latency
  - (Optional) citation hit + semantic similarity when Gemini quota allows

---

## Architecture Overview

**Ingestion**
1. Admin uploads PDFs/TXTs to a topic
2. Backend extracts text → chunks (RecursiveCharacterTextSplitter)
3. Embeddings created via MiniLM
4. Upsert vectors into Pinecone under the topic’s namespace

**Query**
1. User selects topic and asks a question
2. Intent routing:
   - Small talk → direct response
   - Memory → answers only from chat history
   - Knowledge → retrieval + rerank + RAG answer
3. RAG response returns JSON with answer + citations
4. Postgres stores chat history

---

## Project Structure
# Configurable Multi-Tenant Conversational RAG Platform

A production-style **multi-tenant Retrieval-Augmented Generation (RAG)** platform where **organizations (admins)** can create **role-specific topics/agents**, upload proprietary documents, and give **employees/users** access to chat with **citation-grounded** answers.

The system supports:
- **Per-topic document isolation** using **Pinecone namespaces**
- **Per-user chat history** stored in **PostgreSQL**
- **Hybrid conversational behavior**:
  - Small talk 
  - Memory questions from chat history  (“what was my first question?”)
  - RAG answers grounded in uploaded docs  (with citations)
  - Safe fallback when documents don’t contain the answer 

---

## Key Features

###  Multi-Tenant Topics (Role-Specific Agents)
- Admin creates **topics** like “HR Policy”, “Engineering Wiki”, “Onboarding”, etc.
- Each topic has a **behavior prompt** that defines role, tone, and rules.
- **Employees only see topics they are added to**.

###  Secure Admin + User Modes (Product-Style)
- **Admin mode**:
  - One-time bootstrap “Create First Admin”
  - Admin login via token (`X-Admin-Token`)
  - Create topics, upload documents, add employees to topics
- **User mode**:
  - Login via username
  - Select topic
  - Chat with RAG + memory + fallback

###  Citation-Grounded Answers (RAG)
- Documents are chunked and embedded with **MiniLM**.
- Retrieval via **Pinecone vector search**.
- Results reranked with **Cross-Encoder (ms-marco-MiniLM-L-6-v2)**.
- Final answer includes **citations** (source + page when available).

###  Conversation Memory (PostgreSQL)
- Stores per-user (and per-topic) chat history.
- Supports memory-only questions like:
  - “What was my first question?”
  - “Summarize our conversation”
  - “What did I say last time?”

###  Safe Fallback (When Docs Don’t Contain Answer)
When context doesn’t contain the answer:
> “I don’t know this based on the provided documents. But here is what I know about it: …”

This makes the chatbot useful even when retrieval is empty, while remaining honest.

---

## Tech Stack

**Backend**
- FastAPI
- PostgreSQL (chat history, users, topics, membership, admin sessions)
- Pinecone (vector DB + namespaces per topic)
- LangChain + Gemini (response generation)
- HuggingFace MiniLM embeddings
- Cross-Encoder reranker

**Frontend**
- Streamlit (Admin/User UI)

**Evaluation**
- `eval.py` to compute:
  - Recall@5 / Recall@10
  - MRR
  - Retrieval latency & response latency
  - (Optional) citation hit + semantic similarity when Gemini quota allows

---

## Architecture Overview

**Ingestion**
1. Admin uploads PDFs/TXTs to a topic
2. Backend extracts text → chunks (RecursiveCharacterTextSplitter)
3. Embeddings created via MiniLM
4. Upsert vectors into Pinecone under the topic’s namespace

**Query**
1. User selects topic and asks a question
2. Intent routing:
   - Small talk → direct response
   - Memory → answers only from chat history
   - Knowledge → retrieval + rerank + RAG answer
3. RAG response returns JSON with answer + citations
4. Postgres stores chat history

---

## Project Structure
RAGBOT/
backend/
main.py # FastAPI backend (auth, topics, query pipeline)
admin_auth.py # admin hashing/token/session utilities
create_tables.py # DB schema creation
db.py # DB helpers (if used)
guardrails/ # grounded policy checks
retrieval/ # retrieval utilities (optional)
requirements.txt
frontend/
app.py # Streamlit UI (Admin/User)
eval.py # evaluation script
eval_set*.json # evaluation datasets


---

## Setup Instructions

### 1) Create Python environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt

## **2)Environment Variables**

Create a `.env` file in the project root and add:

```env
# Database
DATABASE_URL=your_postgresql_connection_string

# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# BM25 index path
BM25_PATH=bm25.json

# Gemini
GOOGLE_API_KEY=your_google_api_key

##**3)Create DB tables**
python backend/create_tables.py

##**4)Run backend**
python -m uvicorn backend.main:app --reload --port 8000

##**5)Run frontend**
streamlit run frontend/app.py

##**How to Use (Demo Flow)**
1.Admin Flow

2.Choose Admin

3.If first time → Create First Admin

4.Create a topic + behavior prompt 

5.You have a option to improe the prompt , so the prompt can be improved

6.Upload PDF/TXT documents into that topic

7.Add employees to the topic by username

##**User Flow**
1.Choose Employee/User

2.Login with username

3.Select topic

4.Ask questions:

  *Small talk: “hi”

  *RAG: “What is the perinuclear space?”

  *Memory: “What was my first question?”

##**Evaluation**
**Option A (Retrieval-Only Evaluation)  Most reliable**
source venv/bin/activate
export USER_ID="User ID"
export TOPIC_ID="Topic ID"
export EVAL_SET=eval_set_nucleus.json
export SKIP_QUERY=1
python eval.py

**Option B (Full Pipeline Evaluation: retrieval + answer)**
export SKIP_QUERY=0
export TIMEOUT=180
export ANSWER_DELAY_SEC=5
python eval.py

##Architecture: Detailed architecture  flow is available in docs/architecture/.
## Demo (Screenshots)

All demo screenshots are stored in: `docs/screenshots/`

This section explains what each screenshot shows (in the same order as the filenames):

1. **01_landing.png** — App landing page where the user chooses **Admin** vs **Employee/User** mode.

2. **02_admin login page.png** — Admin login screen used to access protected admin actions.

3. **03_admin dashboard 1.png** — Admin dashboard overview showing the main workflow sections:
   - Create Topic
   - Select Topic
   - Upload Documents
   - Add Employee/User to Topic

4. **04_admin dashboard 2 (creating topic).png** — Creating a new topic by entering:
   - topic name  
   - draft topic behavior prompt

5. **05_admin dashboard 3(creating topic).png** — Topic successfully created and visible in the topic selection list.

6. **06_admin dashboard 4 (improving prompt).png** — Using **Improve Prompt** to auto-generate a stronger topic behavior prompt (grounding + citations + safety).

7. **07_admin dashboard (Uploading Documents).png** — Uploading PDF/TXT documents into the selected topic. Files are chunked, embedded, and stored in Pinecone under the topic namespace.

8. **08_User login Page.png** — Employee/User login page (creates a user and loads only accessible topics).

9. **09_User Chat with Ragbot.png** — User chat experience inside a topic:
   - grounded answers when docs contain info  
   - fallback response when the topic docs don’t contain the answer

10. **10_Pinecone dashboard 1.png** — Pinecone index dashboard showing the vector index and record counts.

11. **11_Pinecone namespace.png** — Pinecone namespace view confirming **topic-level isolation** (each topic has its own namespace).

12. **12_Evaluation Results 1(Retrieval only).png** — Evaluation output showing retrieval quality metrics:
   - Recall@5
   - Recall@10
   - MRR

13. **13_Evaluation Results 2(Retrieval only).png** — Additional retrieval-only evaluation output (more test cases / full run).

14. **14_Evaluation Results (retrieval +Semantic Similarity).png** — Extended evaluation output including:
   - retrieval metrics (Recall@K, MRR)
   - semantic similarity (answer vs expected answer)
