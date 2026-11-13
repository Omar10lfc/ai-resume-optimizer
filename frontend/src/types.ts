export interface Artifact {
  kind: "resume-pdf" | "cover-pdf" | "resume-docx" | "resume-tex";
  filename: string;
  url: string;
}

/** MR2: one row of the deterministic requirement fit table (explained in code, never by a model). */
export interface RequirementMatch {
  requirement: string;
  status: "covered" | "partial" | "gap";
  evidence: string[];
  missing: string[];
  reason: string;
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
  /** Live run 22e943c6: the same scan for the COVER letter - fabricated
   *  numbers the letter coined that no input contains. */
  cover_letter_metrics?: string[];
  /** QW2: requests the writer declined because the evidence doesn't back them. */
  declined_requests?: string[];
  /** QW1: per-node prompt fingerprints (e.g. { improver: "improver@2" }). */
  prompt_versions?: Record<string, string>;
  /** MR2: deterministic requirement fit rows (covered/partial/gap + reasons). */
  requirement_matches?: RequirementMatch[];
  /** BB1 Phase B (PB3): the PARSED original resume, echoed server-side when
   *  the request opts in (include_diff=original) - powers DiffView for
   *  uploads; paste mode falls back to the client-held text. */
  original_resume?: string;
  /** MR5: per-task token accounting, keys matching the prompt_versions
   *  stamps (scanner/improver/reviewer/ats-semantic/polisher/cover-letter/
   *  interview-prep). Empty when the run produced no per-task attribution
   *  (fakes that bypass the callback layer). */
  task_usage?: Record<
    string,
    { prompt_tokens: number; completion_tokens: number; total_tokens: number; llm_calls: number }
  >;
}

export type PipelineStage =
  | "idle"
  | "step1_running"
  | "step1_review"
  | "step2_running"
  | "step2_done";

/** BB1: one application in the persistent profile store (summary shape). */
export interface ApplicationSummary {
  id: number;
  title: string;
  status: string;
  created_at: number;
  updated_at: number;
  attempts: number;
  best_score: number | null;
  last_run_at: number | null;
}

/** BB1: one recorded optimization attempt under an application. */
export interface ApplicationRun {
  id: number;
  session: string;
  score: number;
  ats_percentage: number;
  llm_score: number;
  review_failed: number;
  created_at: number;
  result: OptimizationResult | null;
}

/** BB1: an application with all of its runs (detail shape). */
export interface ApplicationDetail extends ApplicationSummary {
  job_text: string;
  resume_text: string;
  runs: ApplicationRun[];
}

export interface NodeStatus {
  node: string;
  label: string;
  state: "pending" | "running" | "completed";
}
