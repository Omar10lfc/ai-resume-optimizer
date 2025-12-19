import gradio as gr
from resume_agent import full_app, scanner_app

def step1_analyze(job_input, resume_file_or_text):
    """
    Step 1: Runs ONLY the Scanner to find gaps.
    Output goes to the 'Human Feedback' box for you to edit.
    """
    # 1. Handle File vs Text Input
    resume_path_or_text = ""
    if hasattr(resume_file_or_text, "name"):
        resume_path_or_text = resume_file_or_text.name  # It's a file path
    else:
        resume_path_or_text = resume_file_or_text       # It's raw text

    # 2. Prepare Inputs
    inputs = {
        "job_description": job_input,
        "original_resume": resume_path_or_text,
        # Initialize empty fields required by the state
        "resume_text": "", 
        "job_text": "", 
        "missing_skills": "" 
    }
    
    try:
        # Run the partial "Scanner Graph"
        result = scanner_app.invoke(inputs)
        
        # Return the missing skills list so you can see/edit it
        return result['missing_skills']
        
    except Exception as e:
        return f"Error during scan: {str(e)}"


def step2_optimize(job_input, resume_file_or_text, user_notes):
    """
    Step 2: Runs the Full Optimizer.
    It uses the 'user_notes' (which you edited in the text box) as context.
    """
    # 1. Handle File vs Text Input
    resume_path_or_text = ""
    if hasattr(resume_file_or_text, "name"):
        resume_path_or_text = resume_file_or_text.name
    else:
        resume_path_or_text = resume_file_or_text

    # 2. Prepare Inputs
    # Notice we pass 'user_notes' into 'human_notes'
    inputs = {
        "job_description": job_input,
        "original_resume": resume_path_or_text,
        "human_notes": user_notes,  # <--- CRITICAL: Uses your edited feedback
        
        # Initialize other fields
        "resume_text": "", "job_text": "", "optimized_resume": "", 
        "feedback": "", "score": 0, "iteration": 0
    }
    
    try:
        # Run the "Full Graph" (Improver + Reviewer Loop)
        final_state = full_app.invoke(inputs)
        
        return (
            final_state.get('optimized_resume', "Error generating resume."), 
            f"Score: {final_state.get('score', 0)}/100", 
            final_state.get('feedback', 'No feedback provided.')
        )
        
    except Exception as e:
        return f"Error: {str(e)}", "Error", "Error"


# --- BUILD THE USER INTERFACE ---

with gr.Blocks(title="AI Resume Agent") as demo:
    gr.Markdown("# AI Resume Optimizer")
    gr.Markdown("An Agentic workflow: **Scan** for gaps first, **Edit** the plan, then **Generate** the result.")
    
    with gr.Row():
        # --- LEFT COLUMN (Inputs) ---
        with gr.Column(scale=1):
            job_in = gr.Textbox(
                label="1. Job Description (URL or Text)", 
                placeholder="Paste Job Text OR a Link (https://...)",
                lines=3
            )
            
            resume_in = gr.File(
                label="2. Upload Resume (PDF)", 
                file_types=[".pdf"],
                type="filepath"
            )
            
            # Action Button 1
            btn_analyze = gr.Button("Step 1: Find Missing Skills", variant="secondary")
            
            # The Feedback Loop Box
            notes_in = gr.Textbox(
                label="3. Missing Skills & Context (Editable)", 
                placeholder="Click 'Step 1' to see missing skills here. \nThen DELETE skills you don't have, or ADD context like 'I used this in my internship...'",
                lines=10, 
                interactive=True
            )
            
            # Action Button 2
            btn_optimize = gr.Button("Step 2: Generate Optimized Resume", variant="primary")
        
        # --- RIGHT COLUMN (Results) ---
        with gr.Column(scale=1):
            score_out = gr.Label(label="Match Score")
            feedback_out = gr.Textbox(label="Critique Summary", lines=4)
            resume_out = gr.Textbox(
                label="Final Optimized Resume", 
                lines=25,
            )

    # --- WIRING THE BUTTONS ---
    
    # 1. Analyze Button
    btn_analyze.click(
        fn=step1_analyze, 
        inputs=[job_in, resume_in], 
        outputs=[notes_in]
    )
    
    # 2. Optimize Button -> Reads 'notes_in' -> Updates final outputs
    btn_optimize.click(
        fn=step2_optimize, 
        inputs=[job_in, resume_in, notes_in], 
        outputs=[resume_out, score_out, feedback_out]
    )

# --- LAUNCH ---
if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())