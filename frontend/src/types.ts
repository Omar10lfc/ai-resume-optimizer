export interface Artifact {
  kind: "resume-pdf" | "cover-pdf" | "resume-docx" | "resume-tex";
  filename: string;
  url: string;
}

export interface OptimizationResult {
  session: string;
  optimized_resume: string;
  cover_letter: string;
  interview_questions: string;
  ats_result: string;
  feedback: string;
  ats_percentage: number;
  score: number;
  score_class: "good" | "warn" | "bad";
  weights: {
    ats: number;
    llm: number;
  };
  artifacts: Artifact[];
  status: string;
}

export type PipelineStage =
  | "idle"
  | "step1_running"
  | "step1_review"
  | "step2_running"
  | "step2_done";

export interface NodeStatus {
  node: string;
  label: string;
  state: "pending" | "running" | "completed";
}
