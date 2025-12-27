import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from dotenv import load_dotenv
import markdown
from xhtml2pdf import pisa
# 1. CONFIGURATION
load_dotenv()

if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = "ENTER_KEY_HERE"

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ==========================================
# 2. STATE DEFINITION
# ==========================================
class AgentState(TypedDict):
    job_description: str
    original_resume: str
    resume_text: str
    job_text: str
    optimized_resume: str 
    feedback: str         
    missing_skills: str   
    human_notes: str
    score: int            
    iteration: int
    cover_letter: str
    resume_pdf_path: str
    cover_letter_pdf_path: str

# ==========================================
# 3. NODE DEFINITIONS
# ==========================================

def loader_node(state: AgentState):
    """NODE 0: Loads PDF or Web data."""
    print(f"\n--- NODE 0: LOADING DATA ---")
    
    # 1. Load Job Description
    job_content = state['job_description']
    if state['job_description'].startswith("http"):
        try:
            loader = WebBaseLoader(state['job_description'])
            docs = loader.load()
            job_content = docs[0].page_content[:5000]
        except Exception as e:
            print(f"   [Error Loading URL]: {e}")
    
    # 2. Load Resume
    resume_content = state['original_resume']
    if state['original_resume'].endswith(".pdf"):
        try:
            loader = PyPDFLoader(state['original_resume'])
            pages = loader.load()
            resume_content = "\n".join([p.page_content for p in pages])
        except Exception as e:
            print(f"   [Error Loading PDF]: {e}")

    return {"job_text": job_content, "resume_text": resume_content}


def scanner_node(state: AgentState):
    """NODE 1: Analyzes GAPS."""
    print(f"\n--- NODE 1: SCANNING FOR GAPS ---")
    
    prompt = f"""
    Compare the Resume to the Job Description.
    Identify the 3 biggest MISSING SKILLS or Keywords.
    
    Job: {state['job_text'][:3000]}
    Resume: {state['resume_text'][:3000]}
    
    Return ONLY a bulleted list of the missing skills.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"   Identified Gaps:\n{response.content}")
    
    return {"missing_skills": response.content, "iteration": 0}


def improver_node(state: AgentState):
    """NODE 2: Rewrites resume using Human Notes."""
    current_iter = state.get("iteration", 0) + 1
    print(f"\n--- NODE 2: IMPROVING RESUME (Iteration {current_iter}) ---")
    
    base_content = state['optimized_resume'] if state['optimized_resume'] else state['resume_text']
    
    prompt = f"""
    You are an expert Resume Writer.
    
    TASK: Rewrite the resume to match the Job Description.
    Output the result in clean MARKDOWN format (use ## for headers, * for bullets).
    
    CRITICAL INSTRUCTIONS:
    1. Address these missing skills: {state['missing_skills']}
    2. USE THIS USER CONTEXT: "{state['human_notes']}" (Incorporate this experience if valid).
    3. If the user provided NO context for a missing skill, do NOT explicitly list it with tags like "(no experience)".
    4. Instead, if it is a specific tool (e.g., "LlamaIndex") and they have no experience, OMIT IT entirely.
    5. If it is a general concept (e.g., "CI/CD" or "MLOps") and they have a CS degree, you may use phrasing like "Conceptual Knowledge of...".
    6. Do NOT invent false projects.
    7. Feedback from previous review (if any): {state.get('feedback', 'None')}
    
    Job Description: {state['job_text']}
    Current Resume: {base_content}
    
    Return ONLY the rewritten resume text in MARKDOWN.
    """
    
    writer_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
    response = writer_llm.invoke([HumanMessage(content=prompt)])
    
    return {"optimized_resume": response.content, "iteration": current_iter}


class ReviewOutput(BaseModel):
    score: int = Field(description="Score between 0 and 100")
    feedback: str = Field(description="One sentence of specific advice to improve the score")

def reviewer_node(state: AgentState):
    """NODE 3: Strict Scoring."""
    print(f"\n--- NODE 3: REVIEWING DRAFT ---")
    
    structured_llm = llm.with_structured_output(ReviewOutput)
    
    prompt = f"""
    Rate this resume match (0-100) for the job.
    Job: {state['job_text']}
    Resume: {state['optimized_resume']}
    """
    
    result = structured_llm.invoke([HumanMessage(content=prompt)])
    
    print(f"-> Score: {result.score}/100")
    print(f"-> Feedback: {result.feedback}")
    
    return {"score": result.score, "feedback": result.feedback}

# --- DOC GENERATION NODES ---

def cover_letter_node(state: AgentState):
    """NODE 4: Generates a tailored Cover Letter."""
    print(f"\n--- NODE 4: GENERATING COVER LETTER ---")
    prompt = f"""
    Write a professional Cover Letter for this Job Description based on the Candidate's Resume.
    Output in clean MARKDOWN format.
    
    Job: {state['job_text'][:2000]}
    Candidate Resume: {state['optimized_resume']}
    
    Keep it concise, engaging, and highlight the key matches found in the resume.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"cover_letter": response.content}


def helper_html_to_pdf(markdown_text, filename, title):
    """
    Converts Markdown -> HTML -> PDF with CSS Styling.
    """
    # 1. Your Custom CSS
    css_style = """
    body { font-family: 'Helvetica', sans-serif; font-size: 12px; line-height: 1.4; color: #333; }
    h1 { font-size: 24px; border-bottom: 2px solid #333; margin-bottom: 10px; text-transform: uppercase; }
    h2 { font-size: 16px; border-bottom: 1px solid #ccc; margin-top: 20px; text-transform: uppercase; color: #555; }
    ul { padding-left: 20px; }
    li { margin-bottom: 5px; }
    strong { color: #000; }
    """
    
    # 2. Convert MD to HTML
    html_body = markdown.markdown(markdown_text)
    
    # 3. Construct Full HTML Page
    full_html = f"""
    <html>
    <head>
        <style>{css_style}</style>
    </head>
    <body>
        <h1>{title}</h1>
        {html_body}
    </body>
    </html>
    """
    
    # 4. Save PDF
    with open(filename, "wb") as pdf_file:
        pisa.CreatePDF(full_html, dest=pdf_file)
    
    return filename

def pdf_exporter_node(state: AgentState):
    """
    NODE 5: Saves the Resume and Cover Letter as Styled PDFs.
    """
    print(f"\n--- NODE 5: EXPORTING PDFs ---")
    
    resume_file = "Optimized_Resume.pdf"
    cover_file = "Cover_Letter.pdf"
    
    # Generate Resume PDF
    helper_html_to_pdf(state['optimized_resume'], resume_file, "Resume")
    
    # Generate Cover Letter PDF
    helper_html_to_pdf(state['cover_letter'], cover_file, "Cover Letter")
    
    print(f"-> Saved: {resume_file}")
    print(f"-> Saved: {cover_file}")
    
    return {
        "resume_pdf_path": resume_file,
        "cover_letter_pdf_path": cover_file
    }
# ==========================================
# 4. GRAPH CONSTRUCTION
# ==========================================

def should_continue(state: AgentState):
    if state['score'] >= 85:
        print("Success! Score is high enough. Generating Docs...")
        return "generate_docs"
    elif state['iteration'] >= 3:
        print("Max retries reached. Generating Docs anyway...")
        return "generate_docs"
    else:
        print("Score too low. Retrying...")
        return "retry"

# --- GRAPH A: FULL OPTIMIZER ---
workflow = StateGraph(AgentState)
workflow.add_node("loader", loader_node)
workflow.add_node("scanner", scanner_node)
workflow.add_node("improver", improver_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("cover_letter", cover_letter_node)
workflow.add_node("pdf_exporter", pdf_exporter_node)

workflow.set_entry_point("loader")
workflow.add_edge("loader", "scanner")
workflow.add_edge("scanner", "improver") 
workflow.add_edge("improver", "reviewer")

workflow.add_conditional_edges(
    "reviewer",
    should_continue,
    {
        "generate_docs": "cover_letter",
        "retry": "improver"
    }
)

workflow.add_edge("cover_letter", "pdf_exporter")
workflow.add_edge("pdf_exporter", END)

full_app = workflow.compile()


# --- GRAPH B: SCANNER ONLY ---
scan_workflow = StateGraph(AgentState)
scan_workflow.add_node("loader", loader_node)
scan_workflow.add_node("scanner", scanner_node)
scan_workflow.set_entry_point("loader")
scan_workflow.add_edge("loader", "scanner")
scan_workflow.add_edge("scanner", END)
scanner_app = scan_workflow.compile()

# ==========================================
# 5. CLI EXECUTION
# ==========================================
if __name__ == "__main__":
    sample_job = "Looking for a Python Developer with Data Science skills."
    sample_resume = "I am a Python developer."
    sample_notes = "I have 2 years of experience."

    print("Starting Advanced Resume Agent (CLI Mode)...")
    try:
        final_state = full_app.invoke({
            "job_description": sample_job, 
            "original_resume": sample_resume,
            "human_notes": sample_notes,
            "resume_text": "", "job_text": "", "optimized_resume": "",
            "feedback": "", "missing_skills": "", "score": 0, "iteration": 0,
            "cover_letter": "", "resume_pdf_path": "", "cover_letter_pdf_path": ""
        })
        print("Done! Files saved.")
    except Exception as e:
        print(f"Error: {e}")