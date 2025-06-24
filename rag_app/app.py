# rag_app/app.py

import os
import asyncio
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# ─── Load .env ────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your .env")

# ─── Flask setup (templates in rag_app/templates, static in project_root/static) ─
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))  # .../rag_app
PROJECT_ROOT = os.path.dirname(BASE_DIR)                   # project root
TEMPLATES    = os.path.join(BASE_DIR, "templates")
STATIC_DIR   = os.path.join(PROJECT_ROOT, "static")

app = Flask(
    __name__,
    template_folder=TEMPLATES,
    static_folder=STATIC_DIR,
)

# ─── 1) Ingest PDFs if missing ─────────────────────────────
from rag_app.ingest import ingest_all
ingest_all()

# ─── 2) Build FAISS + RetrievalQA chains ───────────────────
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

emb = OpenAIEmbeddings(openai_api_key=API_KEY)
llm = ChatOpenAI(openai_api_key=API_KEY, temperature=0)

oncology_vs  = FAISS.load_local( os.path.join(PROJECT_ROOT,"oncology_index"),  emb, allow_dangerous_deserialization=True )
neurology_vs = FAISS.load_local( os.path.join(PROJECT_ROOT,"neurology_index"), emb, allow_dangerous_deserialization=True )

retrieval_qa_oncology  = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=oncology_vs.as_retriever(search_kwargs={"k":3}),
    return_source_documents=True,
)




retrieval_qa_neurology = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=neurology_vs.as_retriever(search_kwargs={"k":3}),
    return_source_documents=True,
    verbose=True,
)


# ─── 3) Unified RAG function ───────────────────────────────

async def rag_qa(question: str) -> str:
    onc_res = retrieval_qa_oncology({'query': question})
    neuro_res = retrieval_qa_neurology({'query': question})

    onc_answer = onc_res['result']
    neuro_answer = neuro_res['result']

    # Heuristic: prefer non-empty, informative answers, avoid defaults like "I don't know"
    def is_informative(answer):
        if not answer: return False
        lower = answer.lower()
        return not any(x in lower for x in ["i don't have", "no information", "not available", "based on the provided context"])

    if is_informative(neuro_answer) and not is_informative(onc_answer):
        return neuro_answer
    elif is_informative(onc_answer) and not is_informative(neuro_answer):
        return onc_answer
    elif is_informative(neuro_answer) and is_informative(onc_answer):
        # Return longer answer assuming it has more info
        return neuro_answer if len(neuro_answer) > len(onc_answer) else onc_answer
    else:
        return "Sorry, I couldn't find relevant information in either Oncology or Neurology documents."

    # Query both indexes
    onc_res = retrieval_qa_oncology({"query": question})
    neu_res = retrieval_qa_neurology({"query": question})
    onc_docs = onc_res.get("source_documents", [])
    neu_docs = neu_res.get("source_documents", [])

    # Pick whichever index returned more hits (tie → oncology)
    if len(onc_docs) >= len(neu_docs) and onc_docs:
        return onc_res.get("result") or onc_res.get("answer", "")
    elif neu_docs:
        return neu_res.get("result") or neu_res.get("answer", "")
    # true fallback
    return "I don't have that information."

def get_answer(question: str) -> str:
    return asyncio.run(rag_qa(question))

# ─── 4) Flask routes ───────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    q = request.get_json(force=True).get("question", "").strip()
    ans = get_answer(q)
    return jsonify({"answer": ans})

if __name__ == "__main__":
    app.run(debug=True)
