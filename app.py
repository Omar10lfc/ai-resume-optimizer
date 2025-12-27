import gradio as gr
import os
from resume_agent import full_app, scanner_app

def step1_analyze(job_input, resume_file_or_text):
    """
    Step 1: Runs ONLY the Scanner to find gaps.
    Output goes to the 'Human Feedback' box for you to edit.
    """
    # Handle File vs Text Input
    resume_path_or_text = ""
    if hasattr(resume_file_or_text, "name"):
        resume_path_or_text = resume_file_or_text.name
    else:
        resume_path_or_text = resume_file_or_text

    # Prepare Inputs
    inputs = {
        "job_description": job_input,
        "original_resume": resume_path_or_text,
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
    Step 2: Runs the Full Optimizer + Document Generator.
    It returns text previews AND file paths for the PDFs.
    """
    # 1. Handle File vs Text Input
    resume_path_or_text = ""
    if hasattr(resume_file_or_text, "name"):
        resume_path_or_text = resume_file_or_text.name
    else:
        resume_path_or_text = resume_file_or_text

    # 2. Prepare Inputs
    inputs = {
        "job_description": job_input,
        "original_resume": resume_path_or_text,
        "human_notes": user_notes,  # Uses your edited feedback
        
        # Initialize empty fields for the state
        "resume_text": "", "job_text": "", "optimized_resume": "", 
        "feedback": "", "score": 0, "iteration": 0,
        "cover_letter": "", "resume_pdf_path": "", "cover_letter_pdf_path": ""
    }
    
    try:
        # Run the "Full Graph" (Improver -> Reviewer -> PDF Gen)
        final_state = full_app.invoke(inputs)
        
        # Extract Results
        opt_resume_text = final_state.get('optimized_resume', "Error generating resume.")
        score_text = f"Score: {final_state.get('score', 0)}/100"
        feedback_text = final_state.get('feedback', 'No feedback provided.')
        
        # Get PDF Paths
        resume_pdf = final_state.get('resume_pdf_path')
        cover_pdf = final_state.get('cover_letter_pdf_path')
        
        return opt_resume_text, score_text, feedback_text, resume_pdf, cover_pdf
        
    except Exception as e:
        return f"Error: {str(e)}", "Error", "Error", None, None


# --- BUILD THE USER INTERFACE ---

with gr.Blocks(title="AI Resume Agent") as demo:
    gr.Markdown("# AI Resume Optimizer")
    gr.Markdown("An Agentic workflow: **Scan** for gaps, **Edit** the plan, then **Generate** professional PDFs.")
    
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
            
            notes_in = gr.Textbox(
                label="3. Missing Skills & Context (Editable)", 
                placeholder="Click 'Step 1' to see missing skills here. \nThen DELETE skills you don't have, or ADD context like 'I used this in my internship...'",
                lines=10, 
                interactive=True
            )
            
            # Action Button 2
            btn_optimize = gr.Button("Step 2: Generate Optimized Resume", variant="primary")
        
        # --- RIGHT COLUMN (Outputs) ---
        with gr.Column(scale=1):
            score_out = gr.Label(label="Match Score")
            feedback_out = gr.Textbox(label="Critique Summary", lines=4)
            
            # Text Preview - REMOVED 'show_copy_button' to fix crash
            resume_out = gr.Textbox(
                label="Final Resume (Text Preview)", 
                lines=15
            )
            
            # PDF Download Area
            gr.Markdown("### 📥 Download Documents")
            with gr.Row():
                pdf_resume_out = gr.File(label="Optimized Resume PDF")
                pdf_cover_out = gr.File(label="Cover Letter PDF")

    # --- WIRING THE BUTTONS ---
    
    # 1. Analyze Button -> Updates the 'notes_in' box
    btn_analyze.click(
        fn=step1_analyze, 
        inputs=[job_in, resume_in], 
        outputs=[notes_in]
    )
    
    # 2. Optimize Button -> Returns Text + PDF Files
    btn_optimize.click(
        fn=step2_optimize, 
        inputs=[job_in, resume_in, notes_in], 
        outputs=[resume_out, score_out, feedback_out, pdf_resume_out, pdf_cover_out]
    )
    
if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())