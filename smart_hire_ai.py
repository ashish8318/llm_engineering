import json
import fitz  # PyMuPDF
from typing import List
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser

# -----------------------------
# CONFIG
# -----------------------------
JOBS_FILE = "./jobs.json"
RESUME_FILE = "./Ashish Resume.pdf"
VECTOR_DB_PATH = "./jobs_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:1b"

# -----------------------------
# PYDANTIC MODELS
# -----------------------------
class JobMatch(BaseModel):
    job_id: str = Field(description="Unique job ID")
    title: str = Field(description="Job title")
    company: str = Field(description="Company name")
    match_score: int = Field(description="Match score from 0 to 100")
    reason: str = Field(description="Why this job matches the candidate")


class JobMatchList(BaseModel):
    jobs: List[JobMatch]


# -----------------------------
# STEP 1: Load Jobs JSON
# -----------------------------
with open(JOBS_FILE, "r", encoding="utf-8") as file:
    jobs = json.load(file)

# -----------------------------
# STEP 2: Convert Jobs to Documents
# -----------------------------
job_docs = []

for job in jobs:
    content = f"""
Job ID: {job['job_id']}
Job Title: {job['title']}
Company: {job['company']}
Location: {job['location']}
Employment Type: {job['employment_type']}
Experience Required: {job['experience_required']}
Salary Range: {job['salary_range']}
Skills Required: {', '.join(job['skills_required'])}
Education: {job['education']}
Job Description: {job['job_description']}
Responsibilities: {', '.join(job['responsibilities'])}
Preferred Qualifications: {', '.join(job['preferred_qualifications'])}
"""

    job_docs.append(
        Document(
            page_content=content,
            metadata={
                "job_id": job["job_id"],
                "title": job["title"],
                "company": job["company"]
            }
        )
    )

# -----------------------------
# STEP 3: Embedding Model
# -----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

# -----------------------------
# STEP 4: Store Jobs in ChromaDB
# -----------------------------
vector_db = Chroma.from_documents(
    documents=job_docs,
    embedding=embedding_model,
    persist_directory=VECTOR_DB_PATH
)

# -----------------------------
# STEP 5: Extract Resume Text
# -----------------------------
def extract_text_from_pdf(pdf_path):
    text = ""
    pdf = fitz.open(pdf_path)

    for page in pdf:
        text += page.get_text()

    pdf.close()
    return text


resume_text = extract_text_from_pdf(RESUME_FILE)

# -----------------------------
# STEP 6: Retrieve Top Jobs
# -----------------------------
retriever = vector_db.as_retriever(search_kwargs={"k": 5})
similar_jobs = retriever.invoke(resume_text)

retrieved_jobs_text = "\n\n".join(
    [job.page_content for job in similar_jobs]
)

# -----------------------------
# STEP 7: Initialize LLM
# -----------------------------
llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0
)

# -----------------------------
# STEP 8: Setup Output Parser
# -----------------------------
parser = PydanticOutputParser(pydantic_object=JobMatchList)

# -----------------------------
# STEP 9: Prompt
# -----------------------------
prompt = f"""
You are an AI recruitment assistant.

Analyze the candidate resume against the provided jobs and rank the best matching jobs.

{parser.get_format_instructions()}

Resume:
{resume_text}

Jobs:
{retrieved_jobs_text}
"""

# -----------------------------
# STEP 10: Invoke LLM
# -----------------------------
response = llm.invoke(prompt)

# -----------------------------
# STEP 11: Parse Output
# -----------------------------
try:
    parsed_result = parser.parse(response.content)

    print("\nFinal AI Matched Jobs:\n")

    for idx, job in enumerate(parsed_result.jobs, start=1):
        print(f"Match #{idx}")
        print("Job ID:", job.job_id)
        print("Title:", job.title)
        print("Company:", job.company)
        print("Match Score:", job.match_score)
        print("Reason:", job.reason)
        print("-" * 50)

except Exception as e:
    print("Parsing Error:", str(e))
    print("Raw LLM Output:")
    print(response.content)