# app.py
import os
import json
import streamlit as st
import fitz  # PyMuPDF
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import random
from transformers import pipeline
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
from langchain_text_splitters import CharacterTextSplitter
import numpy as np
import math
import time

# ---------- Utilities ----------
def extract_text_from_pdf(uploaded_file) -> str:
    file_bytes = uploaded_file.read()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = "\n".join(page.get_text("text") for page in doc)
    return text.strip()
   
  # ===== Offline HuggingFace + LangChain RAG (drop-in) =====

# --- Models & resources (choose small models for CPU demos) ---
GEN_MODEL_NAME = "google/flan-t5-small"         # text generation (seq2seq)
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # embeddings

# --- Initialize generator pipeline (transformers) ---
# Tokenizer + model
gen_pipeline = pipeline(
    "text2text-generation",
    model=GEN_MODEL_NAME,
    device=-1,           # force CPU
    max_length=256       # prevents memory crash
)

# --- Initialize embeddings model ---
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# --- Helper: chunk text ---
text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)

def build_faiss_index(docs):
    """
    docs: list of strings (text chunks)
    returns (index, embeddings, docs)
    """
    embeddings = embed_model.encode(docs, convert_to_numpy=True, show_progress_bar=False)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index, embeddings, docs

def retrieve_topk(index, embeddings, docs, query, k=4):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(np.array(q_emb).astype("float32"), k)
    indices = I[0].tolist()
    results = []
    for idx in indices:
        if idx < len(docs):
            results.append(docs[idx])
    return results

def contextual_prompt_for_questions(context_chunks, n_questions):
    """
    Build a compact prompt that the generator understands.
    Keep it short to avoid long outputs.
    """
    # Join a few retrieved chunks
    ctx = "\n".join(context_chunks)
    prompt = f"""
You are an interview coach. Based on the candidate context and job description below, generate {n_questions} interview items as a JSON array.
Candidate & Job context:
{ctx}

For each item include:
- id (T#, R#, B#)
- category ("Technical"|"Role-specific"|"Behavioral")
- question (short)
- topic (one tag)
- difficulty ("easy"|"medium"|"hard")
- ideal_answer (one short sentence)

Return ONLY valid JSON.
"""
    return prompt

def extract_skills(text: str):
    """
    Very simple skill extractor using keyword matching.
    This avoids any LLM dependency and works offline.
    """
    SKILL_DB = [
        "python", "java", "c", "c++", "react", "node", "express", "django", "flask",
        "machine learning", "deep learning", "nlp", "data science",
        "sql", "mongodb", "postgresql",
        "docker", "kubernetes", "aws", "azure", "gcp",
        "html", "css", "javascript", "typescript",
        "tensorflow", "pytorch",
        "git", "github", "linux"
    ]

    text_lower = text.lower()
    found_skills = [skill for skill in SKILL_DB if skill in text_lower]

    return list(set(found_skills))  # remove duplicates

def call_local_generator(prompt, max_length=256, num_return_sequences=1):
    # Adjust generation settings for deterministic concise outputs
    res = gen_pipeline(prompt, max_length=max_length, do_sample=False, num_return_sequences=num_return_sequences)
    # res is a list of dicts with 'generated_text'
    return res[0]["generated_text"]

def generate_questions_offline(resume_text: str, jd_text: str, n_questions: int = 6):
    """
    High-level function that:
    1) Chunks resume + JD
    2) Builds FAISS index on chunks
    3) Retrieves relevant chunks for combined context
    4) Calls local generator to produce JSON questions
    Returns: (parsed_list_or_raw_text, retrieval_score_estimate)
    """
    # 1) Prepare docs (chunk resume + jd)
    resume_chunks = text_splitter.split_text(resume_text)
    jd_chunks = text_splitter.split_text(jd_text)
    docs = resume_chunks + jd_chunks

    if not docs:
        # fallback simple template
        fallback = """
[
  {"id":"T1","category":"Technical","question":"Explain a core skill listed in your resume.","topic":"CoreSkill","difficulty":"easy","ideal_answer":"Provide a concise explanation and a use-case."}
]
"""
        return (json.loads(fallback), 0.0)

    # 2) Build FAISS index
    index, embeddings, doc_texts = build_faiss_index(docs)

    # 3) Build a short query from JD + resume top words
    # Use a simple heuristic: top noun chunks from the JD (or fallback)
    query = jd_text[:1000] if len(jd_text) > 0 else resume_text[:1000]

    # 4) Retrieve topk chunks (k scaled by n_questions)
    k = min(6, max(3, math.ceil(n_questions * 1.5)))
    retrieved = retrieve_topk(index, embeddings, doc_texts, query, k=k)

    # 5) Build prompt using retrieved context
    prompt = contextual_prompt_for_questions(retrieved, n_questions)

    # 6) Call generator
    try:
        raw_output = call_local_generator(prompt, max_length=512)
    except Exception as e:
        # If generator fails, produce rule-based fallback questions
        # Simple deterministic fallback (non-LLM)
        fallback_questions = []
        skills = extract_skills(resume_text + " " + jd_text)  # reuse your offline extractor if present
        if not skills:
            skills = ["programming", "databases"]
        cnt = 1
        for s in skills[:max(1, n_questions//2)]:
            fallback_questions.append({
                "id": f"T{cnt}",
                "category": "Technical",
                "question": f"Explain how you used {s} in your project.",
                "topic": s,
                "difficulty": "medium",
                "ideal_answer": f"Describe architecture and one example use."
            })
            cnt += 1
        # fill remaining with behavioral templates
        while len(fallback_questions) < n_questions:
            fallback_questions.append({
                "id": f"B{cnt}",
                "category": "Behavioral",
                "question": "Describe a challenge you faced and how you solved it.",
                "topic": "Behavioral",
                "difficulty": "easy",
                "ideal_answer": "Use STAR (situation, task, action, result)."
            })
            cnt += 1
        return fallback_questions, 0.0

    # 7) Try to parse JSON from raw_output
    parsed = None
    try:
        parsed = json.loads(raw_output)
    except Exception:
        # Attempt to extract JSON substring
        start = raw_output.find("[")
        end = raw_output.rfind("]") + 1
        if start != -1 and end != -1:
            try:
                parsed = json.loads(raw_output[start:end])
            except Exception:
                parsed = None

    # 8) If parsing failed, fall back to a mixed strategy: small rule-based set
    if parsed is None:
        # return raw_output for debugging + a small automatic set
        fallback = []
        skills = extract_skills(resume_text + " " + jd_text)
        if not skills:
            skills = ["programming", "databases"]
        cnt = 1
        for s in skills[:max(1, n_questions//2)]:
            fallback.append({
                "id": f"T{cnt}",
                "category": "Technical",
                "question": f"Explain how you used {s} in your project.",
                "topic": s,
                "difficulty": "medium",
                "ideal_answer": f"Describe architecture and one example use."
            })
            cnt += 1
        while len(fallback) < n_questions:
            fallback.append({
                "id": f"B{cnt}",
                "category": "Behavioral",
                "question": "Describe a challenge you faced and how you solved it.",
                "topic": "Behavioral",
                "difficulty": "easy",
                "ideal_answer": "Use STAR method."
            })
            cnt += 1
        # include raw_output for debugging
        return (fallback, 0.0)

    # 9) Compute a simple retrieval score: avg similarity between query and retrieved
    # We'll approximate by embedding the query and computing cosine similarity with retrieved chunk embeddings
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    retrieved_embs = embed_model.encode(retrieved, convert_to_numpy=True)
    # cosine similarity
    def cosine(a,b):
        return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-10))
    similarities = [cosine(q_emb[0], e) for e in retrieved_embs]
    retrieval_score = float(sum(similarities)/len(similarities))*100.0 if similarities else 0.0

    return parsed, retrieval_score

# ===== End offline RAG block =====



def safe_parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end])
            except Exception:
                pass
    return None

# ---------- Streamlit UI ----------
st.set_page_config(page_title="JD2Prep - Interview Question Generator", layout="wide")
st.title("JD2Prep — Resume + JD → AI Interview Preparation")
st.markdown("Upload your **Resume** and the **Job Description** (PDF only). The system will generate customized interview questions with model answers.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Resume (PDF)")
    resume_file = st.file_uploader("Resume PDF", type=["pdf"])
    resume_text = ""
    if resume_file:
        resume_text = extract_text_from_pdf(resume_file)
        st.success("Resume extracted successfully")
        st.text_area("Resume Preview", resume_text, height=220)

with col2:
    st.subheader("Upload Job Description (PDF)")
    jd_file = st.file_uploader("JD PDF", type=["pdf"])
    jd_text = ""
    if jd_file:
        jd_text = extract_text_from_pdf(jd_file)
        st.success("JD extracted successfully")
        st.text_area("JD Preview", jd_text, height=220)

st.markdown("---")
n_questions = st.selectbox("Number of questions", [6, 8, 10, 12], index=1)
generate_btn = st.button("🚀 Generate Interview Prep", type="primary")

# ---------- Generation ----------
if generate_btn:
    if not resume_text or not jd_text:
        st.error("Please upload both Resume and Job Description.")
    else:
        with st.spinner("Generating interview questions..."):
            
            try:
                parsed, retrieval_score = generate_questions_offline(resume_text, jd_text, n_questions)
            except Exception as e:
                st.error(f"Failed {e}")
                st.stop()

            st.success(f"Retrieval relevance score (est.): {retrieval_score:.1f}%")
            items = parsed if isinstance(parsed, list) else parsed

            if not parsed:
                st.error("Failed to parse valid JSON from the model.")
            else:
                st.success("✅ Interview questions generated successfully!")

                for item in parsed:
                    st.markdown(f"### {item['id']} — {item['question']}")
                    st.write(f"**Category:** {item['category']}")
                    st.write(f"**Topic:** {item['topic']}")
                    st.write(f"**Difficulty:** {item['difficulty']}")
                    st.info(f"**Ideal Answer:** {item['ideal_answer']}")
                    st.markdown("---")

                out_json = json.dumps(parsed, indent=2)
                st.download_button("Download as JSON", out_json, "interview_prep.json", "application/json")
                st.download_button("Download as TXT", out_json, "interview_prep.txt", "text/plain")


st.caption("• Resume + JD → Interview Prep Generator")
