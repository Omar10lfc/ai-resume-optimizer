import { useState, useEffect, useRef } from "react";
import { Header } from "./components/Header";
import { StepTimeline } from "./components/StepTimeline";
import { InputSection } from "./components/InputSection";
import { HumanReviewModal } from "./components/HumanReviewModal";
import { ResultsDashboard } from "./components/ResultsDashboard";
import { runStep1, runStep2, checkProviderHealth } from "./api";
import type { PipelineStage, OptimizationResult } from "./types";
import { WarningCircle, Warning, X, SlidersHorizontal, CaretDown, CaretUp, FileText } from "@phosphor-icons/react";

export function App() {
  const [healthOk, setHealthOk] = useState<boolean | null>(null);
  const [healthMessage, setHealthMessage] = useState<string | null>(null);
  const [stage, setStage] = useState<PipelineStage>("idle");
  const [statusMessage, setStatusMessage] = useState<string>("");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [warningMessage, setWarningMessage] = useState<string | null>(null);
  // Graph node ids in arrival order, fed by status events carrying a node id
  // (drives the pipeline timeline without prose-sniffing status messages)
  const [nodeLog, setNodeLog] = useState<string[]>([]);

  const [session, setSession] = useState<string>("");
  const [jobMode, setJobMode] = useState<"text" | "url">("text");
  const [jobText, setJobText] = useState<string>("");
  const [jobUrl, setJobUrl] = useState<string>("");

  const [resumeMode, setResumeMode] = useState<"upload" | "paste">("upload");
  const [resumeText, setResumeText] = useState<string>("");
  const [resumeFile, setResumeFile] = useState<File | null>(null);

  const [missingSkills, setMissingSkills] = useState<string>("");
  const [userNotes, setUserNotes] = useState<string>("");

  const [result, setResult] = useState<OptimizationResult | null>(null);
  const [showInputs, setShowInputs] = useState<boolean>(false);

  const resultsRef = useRef<HTMLDivElement>(null);

  // Poll health on mount
  useEffect(() => {
    checkProviderHealth().then((res) => {
      setHealthOk(res.ok);
      setHealthMessage(res.message);
    });
  }, []);

  // Smooth scroll to results once Step 2 finishes
  useEffect(() => {
    if (stage === "step2_done" && result) {
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 150);
    }
  }, [stage, result]);

  const handleReset = () => {
    setStage("idle");
    setStatusMessage("");
    setErrorMessage(null);
    setWarningMessage(null);
    setNodeLog([]);
    setSession("");
    setMissingSkills("");
    setUserNotes("");
    setResult(null);
    setShowInputs(false);
  };

  // Node-bearing statuses append to the nodeLog; generic ones only update the chip
  const handleStreamStatus = (msg: string, node?: string) => {
    setStatusMessage(msg);
    if (node) setNodeLog((log) => [...log, node]);
  };

  const handleAnalyzeStep1 = () => {
    setErrorMessage(null);
    setWarningMessage(null);
    setNodeLog([]);
    setStage("step1_running");
    setStatusMessage("Initializing documents...");

    const fd = new FormData();
    const finalJob = jobMode === "url" ? jobUrl.trim() : jobText.trim();
    fd.append("job_description", finalJob);

    if (resumeMode === "upload" && resumeFile) {
      fd.append("resume_file", resumeFile);
    } else {
      fd.append("resume_text", resumeText.trim());
    }

    if (session) {
      fd.append("session", session);
    }

    runStep1(fd, {
      onStatus: handleStreamStatus,
      onDone: (data) => {
        setSession(data.session);
        setMissingSkills(data.missing_skills);
        setStage("step1_review");
        setStatusMessage("Gap analysis complete: awaiting your review.");
      },
      onError: (err) => {
        setErrorMessage(err);
        setStage("idle");
        setStatusMessage("");
      },
    });
  };

  const handleOptimizeStep2 = () => {
    setErrorMessage(null);
    setWarningMessage(null);
    setNodeLog([]);
    setStage("step2_running");
    setStatusMessage("Resuming pipeline from checkpoint...");

    const fd = new FormData();
    const finalJob = jobMode === "url" ? jobUrl.trim() : jobText.trim();
    fd.append("job_description", finalJob);

    if (resumeMode === "upload" && resumeFile) {
      fd.append("resume_file", resumeFile);
    } else {
      fd.append("resume_text", resumeText.trim());
    }

    fd.append("notes", userNotes);
    if (session) {
      fd.append("session", session);
    }

    runStep2(fd, {
      onStatus: handleStreamStatus,
      onWarning: (msg) => setWarningMessage(msg),
      onSession: (s) => setSession(s),
      onResult: (res) => {
        setSession(res.session);
        setResult(res);
        setStage("step2_done");
        setStatusMessage("Career documents generated successfully.");
      },
      onError: (err) => {
        setErrorMessage(err);
        setStage("step1_review");
        setStatusMessage("");
      },
    });
  };

  const isProcessing = stage === "step1_running" || stage === "step2_running";

  return (
    <div className="min-h-[100dvh] bg-[#090a0f] text-[#f8fafc] flex flex-col font-sans selection:bg-[#06b6d4]/20 selection:text-[#06b6d4]">
      {/* 1. Header */}
      <Header
        healthOk={healthOk}
        healthMessage={healthMessage}
        onReset={handleReset}
        isProcessing={isProcessing}
      />

      {/* 2. Main Content Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 pt-8 pb-16">
        {/* Error notification banner */}
        {errorMessage && (
          <div className="mb-6 p-4 rounded-xl bg-rose-950/40 border border-rose-500/40 text-rose-300 text-sm flex items-center justify-between gap-3 animate-fade-in">
            <div className="flex items-center gap-2.5">
              <WarningCircle weight="bold" className="w-5 h-5 text-rose-400 shrink-0" />
              <span>{errorMessage}</span>
            </div>
            <button
              onClick={() => setErrorMessage(null)}
              className="p-1 hover:text-white transition-colors"
            >
              <X weight="bold" className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* Warning notification banner (mid-stream notices, e.g. auto-detected gaps) */}
        {warningMessage && (
          <div className="mb-6 p-4 rounded-xl bg-amber-950/40 border border-amber-500/40 text-amber-300 text-sm flex items-center justify-between gap-3 animate-fade-in">
            <div className="flex items-center gap-2.5">
              <Warning weight="bold" className="w-5 h-5 text-amber-400 shrink-0" />
              <span>{warningMessage}</span>
            </div>
            <button
              onClick={() => setWarningMessage(null)}
              className="p-1 hover:text-white transition-colors"
            >
              <X weight="bold" className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* Hero Section: Lean, un-bloated copy */}
        <section className="mb-8">
          <div className="max-w-3xl">
            <span className="text-xs font-mono uppercase tracking-wider text-[#06b6d4]">
              Autonomous Career Agent
            </span>
            <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight text-[#f8fafc] mt-2 mb-3 leading-tight">
              Optimize your resume for real job descriptions.
            </h1>
            <p className="text-sm sm:text-base text-[#94a3b8] leading-relaxed max-w-[65ch]">
              Deterministic ATS keyword scoring paired with multi-agent self-refinement and human-in-the-loop checkpoints.
            </p>
          </div>
        </section>

        {/* 3. Pipeline Step Visualizer */}
        <StepTimeline stage={stage} statusMessage={statusMessage} nodeLog={nodeLog} />

        {/* 4. Input Section: Collapsed once results are ready, expandable on demand */}
        {stage === "step2_done" && !showInputs ? (
          <div className="mb-8 p-4 rounded-xl bg-[#10131c] border border-[#1e2436] flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs animate-fade-in">
            <div className="flex flex-wrap items-center gap-2.5">
              <div className="flex items-center gap-1.5 text-[#06b6d4] font-mono">
                <FileText weight="bold" className="w-4 h-4" />
                <span>Target Job:</span>
              </div>
              <span className="text-[#f8fafc] font-medium">
                {jobMode === "url"
                  ? jobUrl || "URL provided"
                  : `${jobText.trim().split(/\s+/).filter(Boolean).length || 0} words`}
              </span>
              <span className="text-[#64748b]">|</span>
              <div className="flex items-center gap-1.5 text-[#06b6d4] font-mono">
                <span>Candidate CV:</span>
              </div>
              <span className="text-[#f8fafc] font-medium">
                {resumeMode === "upload" && resumeFile ? resumeFile.name : `${resumeText.length} characters`}
              </span>
            </div>
            <button
              type="button"
              onClick={() => setShowInputs(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#151926] hover:bg-[#1b2030] text-[#94a3b8] hover:text-[#f8fafc] border border-[#1e2436] font-medium transition-all self-start sm:self-auto shrink-0 active:scale-[0.98]"
            >
              <SlidersHorizontal weight="bold" className="w-3.5 h-3.5 text-[#06b6d4]" />
              <span>Modify Inputs</span>
              <CaretDown weight="bold" className="w-3 h-3 text-[#64748b]" />
            </button>
          </div>
        ) : (
          <div className="relative mb-8">
            {stage === "step2_done" && (
              <div className="flex justify-end mb-2">
                <button
                  type="button"
                  onClick={() => setShowInputs(false)}
                  className="flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-mono text-[#94a3b8] hover:text-[#f8fafc] bg-[#10131c] border border-[#1e2436] transition-all"
                >
                  <span>Collapse Inputs</span>
                  <CaretUp weight="bold" className="w-3 h-3 text-[#64748b]" />
                </button>
              </div>
            )}
            <InputSection
              jobText={jobText}
              setJobText={setJobText}
              jobUrl={jobUrl}
              setJobUrl={setJobUrl}
              jobMode={jobMode}
              setJobMode={setJobMode}
              resumeText={resumeText}
              setResumeText={setResumeText}
              resumeFile={resumeFile}
              setResumeFile={setResumeFile}
              resumeMode={resumeMode}
              setResumeMode={setResumeMode}
              onAnalyze={handleAnalyzeStep1}
              isProcessing={stage === "step1_running"}
              disabled={stage === "step2_running"}
            />
          </div>
        )}

        {/* 5. Human-in-the-Loop Review (When Step 1 has completed) */}
        {(stage === "step1_review" || stage === "step2_running") && (
          <HumanReviewModal
            missingSkills={missingSkills}
            userNotes={userNotes}
            setUserNotes={setUserNotes}
            onSubmit={handleOptimizeStep2}
            isProcessing={stage === "step2_running"}
          />
        )}

        {/* 6. Results Dashboard (When Step 2 has completed) */}
        {result && (
          <div ref={resultsRef} className="scroll-mt-20">
            <ResultsDashboard result={result} />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="w-full border-t border-[#1e2436] py-6 px-4 sm:px-6 bg-[#090a0f] text-xs text-[#64748b] text-center font-mono">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3">
          <span>AI Resume Optimizer : LangGraph State Machine Architecture</span>
          <div className="flex items-center gap-4 text-[#94a3b8]">
            <span>Groq GPT-OSS Inference</span>
            <span>LaTeX PDF Engine</span>
            <span>LangSmith Observability</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
