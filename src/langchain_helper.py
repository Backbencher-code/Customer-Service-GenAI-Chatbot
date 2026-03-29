import os
import logging

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    # model="gemini-2.0-flash-lite",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.2,
)

# ── EMBEDDINGS ────────────────────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ── PATHS ─────────────────────────────────────────────────────────────────────
# APP_BASE_DIR can be overridden via environment variable for flexibility
BASE_DIR = os.environ.get(
    "APP_BASE_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
VECTORDB_PATH = os.path.join(BASE_DIR, "faiss_index")
CSV_PATH = os.path.join(BASE_DIR, "dataset", "dataset.csv")


def create_vector_db() -> int:
    """
    Reads the FAQ CSV, embeds every row, and saves the FAISS index to disk.
    Returns the number of documents indexed.
    """
    loader = CSVLoader(file_path=CSV_PATH, source_column="prompt")
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    vectordb.save_local(VECTORDB_PATH)
    logger.info("Vector DB created with %d documents.", len(data))
    return len(data)


def get_retriever():
    """
    Loads the FAISS index from disk and returns a retriever object.
    Raises FileNotFoundError with a helpful message if the index doesn't exist.
    """
    if not os.path.exists(VECTORDB_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at '{VECTORDB_PATH}'. "
            "Please click 'Build Knowledge Base' in the sidebar first."
        )

    vectordb = FAISS.load_local(
        VECTORDB_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectordb.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5, "k": 3},
    )


def get_qa_chain() -> LLMChain:
    """
    Returns an LLMChain with a prompt that has three placeholders:
    {chat_history}, {context}, {question}.
    Context is manually retrieved and passed from main.py — this gives
    full control and avoids all RetrievalQA key conflicts.
    """
    prompt_template = """You are a friendly and helpful customer support assistant for ShopEase, \
an Indian e-commerce platform. Use ONLY the context below to answer the question.

Rules:
- Be concise but complete.
- If the answer is not found in the context, say: "I don't have that information right now. \
Please contact our support team via Live Chat or call 1800-XXX-XXXX."
- Never make up information.
- Maintain a warm, professional tone.
- When relevant, mention helpful features like My Orders, ShopEase Wallet, or ShopEase Plus.

Previous conversation:
{chat_history}

Context from knowledge base:
{context}

Customer's question: {question}

Helpful Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["chat_history", "context", "question"],
    )

    chain = LLMChain(llm=llm, prompt=PROMPT)
    return chain


def retrieve_docs(retriever, question: str) -> tuple[list, str]:
    """
    Runs the retriever using the modern .invoke() API (replaces deprecated
    get_relevant_documents()). Returns a (docs, context_string) tuple.
    Logs a warning when no documents pass the similarity threshold.
    """
    docs = retriever.invoke(question)

    if not docs:
        logger.warning(
            "No documents met the similarity threshold for query: '%s'. "
            "The LLM will fall back to its default no-info response.",
            question,
        )

    context = "\n\n".join([doc.page_content for doc in docs])
    return docs, context


def format_chat_history(messages: list) -> str:
    """
    Converts the last 3 exchanges (6 messages) from the messages list into a
    plain-text string for the {chat_history} placeholder.
    """
    recent = messages[-6:] if len(messages) > 6 else messages
    lines = []
    for msg in recent:
        role = "Customer" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


# ── QUICK SMOKE TEST ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    count = create_vector_db()
    print(f"Vector DB created with {count} documents.")

    retriever = get_retriever()
    chain = get_qa_chain()

    question = "Do you offer EMI?"
    docs, context = retrieve_docs(retriever, question)

    result = chain.invoke({
        "question": question,
        "chat_history": "",
        "context": context,
    })
    print("Answer:", result["text"])