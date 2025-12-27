from langgraph.graph import StateGraph, END
from typing import TypedDict
class AgentState(TypedDict):
    job_description: str
    original_resume: str

# 2. Define Placeholder Nodes
def loader_node(state): return {}
def scanner_node(state): return {}
def human_feedback_node(state): return {}
def improver_node(state): return {}
def reviewer_node(state): return {}
def cover_letter_node(state): return {}
def pdf_exporter_node(state): return {}

# 3. Define Logic
def should_continue(state):
    return "perfect"

# 4. Build the Unified Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("Loader", loader_node)
workflow.add_node("Scanner_Analyst", scanner_node)
workflow.add_node("Human_Feedback_Loop", human_feedback_node)
workflow.add_node("Resume_Writer", improver_node)
workflow.add_node("Reviewer_Critic", reviewer_node)
workflow.add_node("Cover_Letter_Writer", cover_letter_node)
workflow.add_node("PDF_Generator", pdf_exporter_node)

# Connect Edges (The Pipeline)
workflow.set_entry_point("Loader")
workflow.add_edge("Loader", "Scanner_Analyst")
workflow.add_edge("Scanner_Analyst", "Human_Feedback_Loop")
workflow.add_edge("Human_Feedback_Loop", "Resume_Writer")
workflow.add_edge("Resume_Writer", "Reviewer_Critic")

# The Conditional Logic (The Brain)
workflow.add_conditional_edges(
    "Reviewer_Critic",
    should_continue,
    {
        "retry": "Resume_Writer",
        "perfect": "Cover_Letter_Writer",
        "max_retries": "Cover_Letter_Writer" 
    }
)

# Document Generation Phase
workflow.add_edge("Cover_Letter_Writer", "PDF_Generator")
workflow.add_edge("PDF_Generator", END)

# 5. Generate and Save Image
app = workflow.compile()

print("Generating Architecture Diagram...")
try:
    png_data = app.get_graph().draw_mermaid_png()
    with open("agent_diagram.png", "wb") as f:
        f.write(png_data)

    print("Saved 'agent_diagram.png' - Check your folder!")
except Exception as e:
    print(f"Error: {e}")
    print("Tip: You might need to install: 'pip install graphviz'")