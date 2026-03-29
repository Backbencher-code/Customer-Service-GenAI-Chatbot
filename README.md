# ShopEase Customer Service Chatbot

A RAG-based customer service chatbot for e-commerce, built using LangChain, FAISS, Google Gemini, and Streamlit. The idea was to build something that actually solves a real problem — instead of making customers wait for a support agent, they can just ask the bot and get an instant answer.

---

## What it does

You type a question like "how do I return a product" or "my payment failed but money got deducted" — and the bot finds the most relevant answer from the knowledge base and responds in a conversational way. It also remembers the last few messages so follow-up questions work naturally.

---

## How it works

The core idea is RAG — Retrieval Augmented Generation. Instead of asking the LLM to answer from memory (which leads to hallucinations), we first retrieve the most relevant FAQ from our own dataset using FAISS, then pass that as context to Gemini to generate the final response.

```
User question
    → Embed with all-MiniLM-L6-v2
    → Search FAISS index for top 3 matching FAQs
    → Build prompt with context + chat history
    → Gemini generates the answer
```

One thing worth mentioning — I avoided LangChain's RetrievalQA chain because it throws an input key conflict error when you try to add memory. Instead I built the pipeline manually: FAISS retrieval → context builder → LLMChain. More code but full control over everything.

---

## Dataset

Built a custom dataset of 80 FAQ pairs modelled after real Indian e-commerce platforms like Flipkart and Myntra. Covers orders, payments, returns, delivery, account management, offers, and more. Stored as a simple CSV with prompt and response columns.

I didn't use any existing dataset because most of them are either too generic or have different column structures. Writing it from scratch also meant I understood every row.

---

## Project structure

```
Customer Service GenAI Chatbot/
├── dataset/
│   └── dataset.csv
├── src/
│   ├── langchain_helper.py
│   └── main.py
├── .env
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/Backbencher-code/customer-service-chatbot.git
cd customer-service-chatbot

# Create and activate virtual environment
python -m venv venv_chatbot
venv_chatbot\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Add your Google API key to .env
GOOGLE_API_KEY="your_key_here"

# Run
streamlit run src/main.py
```

On first run, sentence-transformers will download the embedding model (~80MB). After that it's cached.

---

## Usage

1. Click Build Knowledge Base in the sidebar — this embeds all 80 FAQs into FAISS
2. Start asking questions
3. Try follow-up questions to see the memory working

---

## Tech stack

- LLM — Google Gemini 1.5 Flash via langchain-google-genai
- Embeddings — sentence-transformers/all-MiniLM-L6-v2 (runs locally)
- Vector store — FAISS
- Framework — LangChain
- UI — Streamlit

---

## Bugs I ran into

Documenting these because they took time to figure out:

- google-generativeai and langchain-google-genai conflict with each other due to google-ai-generativelanguage version mismatch. Fix: remove google-generativeai from requirements entirely.
- RetrievalQA throws "One input key expected got ['input_documents', 'question']" when memory is added via chain_type_kwargs. Fix: skip RetrievalQA, do retrieval manually.
- Streamlit throws "session_state cannot be modified after widget is instantiated" if you try to clear the input box directly. Fix: rotate the widget key using an input_counter in session state.
- File paths break when running from a different directory. Fix: use os.path.abspath(__file__) to build absolute paths relative to the script location.

---

## Author

Sahil Rohilla — [LinkedIn](https://linkedin.com/in/sahilrohilla) · [GitHub](https://github.com/Backbencher-code)
