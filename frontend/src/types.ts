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
  /** QW3: what the deterministic integrity scrubber removed, with reasons. */
  removals?: string[];
  /** QW3: metric-shaped numbers still in the draft that the evidence lacks. */
  unevidenced_metrics?: string[];
  /** QW2: requests the writer declined because the evidence doesn't back them. */
  declined_requests?: string[];
  /** QW1: per-node prompt fingerprints (e.g. { improver: "improver@2" }). */
  prompt_versions?: Record<string, string>;
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
