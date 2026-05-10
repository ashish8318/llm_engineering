import json
import fitz
from typing import List
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------
# CONFIG
# -----------------------------
JOBS_FILE = "./jobs.json"
RESUME_FILE = "./Ashish Resume.pdf"
VECTOR_DB_PATH = "./jobs_db_advanced"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
LLM_MODEL = "llama3.2:1b"

# -----------------------------
# PYDANTIC MODELS
# -----------------------------
class JobMatch(BaseModel):
    job_id: str = Field(description="Unique job ID")
    title: str = Field(description="Job title")
    company: str = Field(description="Company name")
    match_score: int = Field(description="Match score from 0 to 100")
    matching_skills: List[str] = Field(description="Matching skills")
    missing_skills: List[str] = Field(description="Missing skills")
    reason: str = Field(description="Detailed reason for match")


class JobMatchList(BaseModel):
    jobs: List[JobMatch]


# -----------------------------
# STEP 1: LOAD JOBS
# -----------------------------
with open(JOBS_FILE, "r", encoding="utf-8") as file:
    jobs = json.load(file)

# -----------------------------
# STEP 2: CREATE DOCUMENTS
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
                "company": job["company"],
                "location": job["location"],
                "skills": ", ".join(job["skills_required"]),
                "experience": job["experience_required"]
            }
        )
    )

# -----------------------------
# STEP 3: SPLIT DOCUMENTS
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

split_docs = splitter.split_documents(job_docs)

# -----------------------------
# STEP 4: EMBEDDINGS
# -----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

# -----------------------------
# STEP 5: VECTOR DB
# -----------------------------
vector_db = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding_model,
    persist_directory=VECTOR_DB_PATH
)

# -----------------------------
# STEP 6: PDF EXTRACTION
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
# STEP 7: RETRIEVAL
# -----------------------------
retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 10,
        "fetch_k": 20
    }
)

retrieved_docs = retriever.invoke(resume_text)

# -----------------------------
# STEP 8: DEDUPLICATE JOBS
# -----------------------------
unique_jobs = {}
for doc in retrieved_docs:
    job_id = doc.metadata["job_id"]
    if job_id not in unique_jobs:
        unique_jobs[job_id] = doc

final_jobs = list(unique_jobs.values())[:5]

retrieved_jobs_text = "\n\n".join(
    [doc.page_content for doc in final_jobs]
)

# -----------------------------
# STEP 9: LLM
# -----------------------------
llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0
)

# -----------------------------
# STEP 10: OUTPUT PARSER
# -----------------------------
parser = PydanticOutputParser(pydantic_object=JobMatchList)

# -----------------------------
# STEP 11: PROMPT
# -----------------------------
prompt = f"""
You are an advanced AI recruitment assistant.

Analyze the candidate's resume and compare it against the retrieved jobs.

For each job:
- Assign a match score (0-100)
- Identify matching skills
- Identify missing skills
- Explain why this role is suitable or unsuitable

Return only valid structured JSON.

{parser.get_format_instructions()}

Candidate Resume:
{resume_text}

Retrieved Jobs:
{retrieved_jobs_text}
"""

# -----------------------------
# STEP 12: LLM RESPONSE
# -----------------------------
response = llm.invoke(prompt)

# -----------------------------
# STEP 13: PARSE RESPONSE
# -----------------------------
try:
    parsed_result = parser.parse(response.content)

    print("\n========== FINAL AI JOB MATCHES ==========\n")

    for idx, job in enumerate(parsed_result.jobs, start=1):
        print(f"Match #{idx}")
        print("Job ID:", job.job_id)
        print("Title:", job.title)
        print("Company:", job.company)
        print("Match Score:", job.match_score)
        print("Matching Skills:", ", ".join(job.matching_skills))
        print("Missing Skills:", ", ".join(job.missing_skills))
        print("Reason:", job.reason)
        print("-" * 80)

except Exception as e:
    print("Parsing Error:", str(e))
    print("\nRaw LLM Output:\n")
    print(response.content)