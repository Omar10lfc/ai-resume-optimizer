import { useState, useEffect, useRef } from "react";
import { Header } from "./components/Header";
import { StepTimeline } from "./components/StepTimeline";
import { InputSection, type JobMeta } from "./components/InputSection";
import { HumanReviewModal } from "./components/HumanReviewModal";
import { ResultsDashboard } from "./components/ResultsDashboard";
import { HistoryPanel } from "./components/HistoryPanel";
import { SettingsModal } from "./components/SettingsModal";
import { ErrorState } from "./components/ErrorState";
import { Onboarding } from "./components/Onboarding";
import { ToastStack } from "./components/ToastStack";
import { useToasts } from "./lib/useToasts";
import { isOnboarded } from "./lib/onboarding";
import { parseGapSkills } from "./lib/gaps";
import { ReviewSkeleton, ResultsSkeleton } from "./components/Skeleton";
import { runStep1, runStep2, checkProviderHealth } from "./api";
import { classifyError, type ErrorInfo } from "./lib/errors";
import type { PipelineStage, OptimizationResult } from "./types";
import { FileText, X } from "@phosphor-icons/react";

// Saved-resume restore for the WEB SPA (localStorage). The extension path
// restores via chrome.storage in the mount effect; this helper no-ops there.
function readSavedWebResume(): string {
  try {
    if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
      return ""; // extension flow owns the restore
    }
    return localStorage.getItem("saved_resume_text") || "";
  } catch {
    return "";
  }
}

// Optional optimizer context survives reloads the same way (web path).
function readSavedWebNotes(): string {
  try {
    if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
      return "";
    }
    return localStorage.getItem("pending_optimizer_notes") || "";
  } catch {
    return "";
  }
}

const RESUME_SAVE_DEBOUNCE_MS = 500;

export function App() {
  const [healthOk, setHealthOk] = useState<boolean | null>(null);
  const [healthMessage, setHealthMessage] = useState<string | null>(null);
  const [stage, setStage] = useState<PipelineStage>("idle");
  const [statusMessage, setStatusMessage] = useState<string>("");
  const [errorInfo, setErrorInfo] = useState<ErrorInfo | null>(null);
  const [warningMessage, setWarningMessage] = useState<string | null>(null);
  // Graph node ids in arrival order, fed by status events carrying a node id
  // (drives the pipeline timeline without prose-sniffing status messages)
  const [nodeLog, setNodeLog] = useState<string[]>([]);
  const [lastEventAt, setLastEventAt] = useState<number | null>(null);
  const [failedNode, setFailedNode] = useState<string | null>(null);
  const [whichStepFailed, setWhichStepFailed] = useState<1 | 2 | null>(null);

  const [session, setSession] = useState<string>("");
  const [jobMode, setJobMode] = useState<"text" | "url">("text");
  const [jobText, setJobText] = useState<string>("");
  const [jobUrl, setJobUrl] = useState<string>("");
  const [jobMeta, setJobMeta] = useState<JobMeta | null>(null);

  // Web SPA restore (localStorage): read once as initial state. The extension
  // equivalent restores via chrome.storage in the mount effect below.
  const savedWebResume = readSavedWebResume();
  const [resumeMode, setResumeMode] = useState<"upload" | "paste">(savedWebResume ? "paste" : "upload");
  const [resumeText, setResumeText] = useState<string>(savedWebResume);
  const [resumeFile, setResumeFile] = useState<File | null>(null);

  const [missingSkills, setMissingSkills] = useState<string>("");
  const [userNotes, setUserNotes] = useState<string>(readSavedWebNotes());
  // True after the scan produced gap chips; false when the zero-gap
  // straight-through path skips the review dialog entirely.
  const [reviewOpen, setReviewOpen] = useState<boolean>(false);

  const [result, setResult] = useState<OptimizationResult | null>(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);
  const [showForm, setShowForm] = useState(true);

  const [onboardingOpen, setOnboardingOpen] = useState(false);
  const { toasts, push, dismiss } = useToasts();

  const resultsRef = useRef<HTMLDivElement>(null);

  // Latest resume input (text/mode/file) for storage callbacks, which run
  // after the mount effect ([]) and would otherwise see stale state closures.
  const latestResumeRef = useRef({ text: "", mode: "upload" as "upload" | "paste", file: null as File | null });

  // Becomes true once a meaningful resume text has been persisted, so actively
  // clearing the resume later purges the stale saved copy (without racing the
  // storage.get restore that happens on mount).
  const savedResumeOnceRef = useRef(false);

  // Latest user notes for closures that must not depend on committed state
  // (the zero-gap auto-run path fires from a step-1 stream callback).
  const userNotesRef = useRef(userNotes);
  useEffect(() => {
    userNotesRef.current = userNotes;
  }, [userNotes]);

  // Persist the optimizer context (chrome.storage.local / localStorage),
  // debounced like the resume vault. Purged by Reset, the Settings eraser,
  // and by actively clearing the text.
  const notesStoredOnceRef = useRef(false);
  useEffect(() => {
    const isExtension = typeof chrome !== "undefined" && !!chrome.storage?.local;
    const persist = () => {
      try {
        if (isExtension) {
          chrome.storage.local.set({ pendingUserNotes: userNotes });
        } else {
          localStorage.setItem("pending_optimizer_notes", userNotes);
        }
      } catch {
        // ignore
      }
    };
    const purge = () => {
      try {
        if (isExtension) {
          chrome.storage.local.remove("pendingUserNotes");
        } else {
          localStorage.removeItem("pending_optimizer_notes");
        }
      } catch {
        // ignore
      }
    };

    if (userNotes.trim() !== "") {
      notesStoredOnceRef.current = true;
    } else if (notesStoredOnceRef.current && userNotes === "") {
      notesStoredOnceRef.current = false;
      purge();
      return;
    }
    if (userNotes.trim() === "") return;

    const t = setTimeout(persist, RESUME_SAVE_DEBOUNCE_MS);
    return () => clearTimeout(t);
  }, [userNotes]);

  useEffect(() => {
    latestResumeRef.current = { text: resumeText, mode: resumeMode, file: resumeFile };
  }, [resumeText, resumeMode, resumeFile]);

  const refreshHealth = () => {
    checkProviderHealth().then((res) => {
      setHealthOk(res.ok);
      setHealthMessage(res.message);
    });
  };

  // Poll health on mount; restore saved inputs; listen for scraped jobs
  useEffect(() => {
    refreshHealth();

    if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
      chrome.storage.local.get(["savedResumeText", "pendingScrapedJob", "onboardedAt", "pendingUserNotes"], (data: Record<string, any>) => {
        const current = latestResumeRef.current;
        if (data.savedResumeText && !current.text && !current.file) {
          setResumeText(data.savedResumeText);
          setResumeMode("paste");
          push("info", "Picked up your saved resume from this browser. Erase it any time in Settings.");
        }
        if (data.pendingUserNotes && !(userNotesRef.current || "").trim()) {
          setUserNotes(data.pendingUserNotes);
          push("info", "Restored your optimization notes from this browser. Erase them in Settings.");
        }
        if (data.pendingScrapedJob) {
          const scraped = data.pendingScrapedJob;
          setJobText(scraped.description || "");
          setJobMode("text");
          setJobMeta({ title: scraped.title || "", company: scraped.company || "" });
          chrome.storage.local.remove(["pendingScrapedJob"]);
        }
        if (!data.onboardedAt) {
          setOnboardingOpen(true);
        }
      });

      const messageListener = (msg: { type?: string; job?: { title: string; company: string; description: string } }) => {
        if (msg.type === "JOB_DETECTED" && msg.job) {
          setJobText(msg.job.description || "");
          setJobMode("text");
          setJobMeta({ title: msg.job.title || "", company: msg.job.company || "" });
        }
      };

      chrome.runtime?.onMessage?.addListener(messageListener);
      return () => {
        chrome.runtime?.onMessage?.removeListener(messageListener);
      };
    }

    // Web SPA: same first-run gate via the flag (extension path handled above)
    if (!(typeof chrome !== "undefined" && chrome.storage && chrome.storage.local)) {
      isOnboarded().then((done) => {
        if (!done) setOnboardingOpen(true);
      });
    }
  }, [push]);

  // Auto-save resume text, debounced: chrome.storage.set per keystroke needs
  // avoiding on long documents (each write re-serializes the whole key).
  useEffect(() => {
    const meaningful = resumeText.trim().length > 50;
    const isExtension = typeof chrome !== "undefined" && !!chrome.storage?.local;

    const persist = (value: string) => {
      try {
        if (isExtension) {
          chrome.storage.local.set({ savedResumeText: value });
        } else {
          localStorage.setItem("saved_resume_text", value);
        }
      } catch {
        // ignore
      }
    };
    const purge = () => {
      try {
        if (isExtension) {
          chrome.storage.local.remove("savedResumeText");
        } else {
          localStorage.removeItem("saved_resume_text");
        }
      } catch {
        // ignore
      }
    };

    if (meaningful) {
      savedResumeOnceRef.current = true;
    } else if (savedResumeOnceRef.current && resumeText.trim() === "") {
      savedResumeOnceRef.current = false;
      purge();
      return; // purged — nothing to save
    }

    if (!meaningful) return;

    const t = setTimeout(() => persist(resumeText), RESUME_SAVE_DEBOUNCE_MS);
    return () => clearTimeout(t);
  }, [resumeText]);

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
    setErrorInfo(null);
    setWarningMessage(null);
    setNodeLog([]);
    setLastEventAt(null);
    setFailedNode(null);
    setWhichStepFailed(null);
    setSession("");
    setMissingSkills("");
    setUserNotes("");
    setReviewOpen(false);
    setResult(null);
    setShowForm(true);
  };

  // Node-bearing statuses append to the nodeLog; generic ones only update the chip
  const handleStreamStatus = (msg: string, node?: string) => {
    setStatusMessage(msg);
    setLastEventAt(Date.now());
    if (node) setNodeLog((log) => [...log, node]);
  };

  const buildFormData = (): FormData => {
    const fd = new FormData();
    const finalJob = jobMode === "url" ? jobUrl.trim() : jobText.trim();
    fd.append("job_description", finalJob);
    if (resumeMode === "upload" && resumeFile) {
      fd.append("resume_file", resumeFile);
    } else {
      fd.append("resume_text", resumeText.trim());
    }
    if (session) fd.append("session", session);
    return fd;
  };

  const handleAnalyzeStep1 = () => {
    setErrorInfo(null);
    setWarningMessage(null);
    setNodeLog([]);
    setFailedNode(null);
    setWhichStepFailed(null);
    setStage("step1_running");
    setStatusMessage("Initializing documents...");
    setLastEventAt(Date.now());
    setShowForm(true);

    runStep1(buildFormData(), {
      onStatus: handleStreamStatus,
      onDone: (data) => {
        setSession(data.session);
        setMissingSkills(data.missing_skills);
        const gapCount = parseGapSkills(data.missing_skills).length;
        const scannerOutput = data.missing_skills.trim().length > 0;
        if (gapCount === 0 && !scannerOutput) {
          // Nothing to decide AND the scanner reported nothing → no review
          // checkpoint. Tailor straight away; optional context typed below
          // the inputs is picked up via the ref.
          setReviewOpen(false);
          setStatusMessage("No missing skills found — optimizing directly.");
          push("info", "No missing skills found — tailoring your resume to this job right away.");
          handleOptimizeStep2();
          return;
        }
        if (gapCount === 0 && scannerOutput) {
          // Scanner produced text the parser can't turn into chips (exotic
          // bullet formatting, prose). Never auto-run past the decision
          // surface — the raw text goes to the review dialog so the user
          // decides what to feature.
          push("info", "Gap scan returned an unusual format — review it below before optimizing.");
        }
        setReviewOpen(true);
        setStage("step1_review");
        setStatusMessage("Gap analysis complete — awaiting your review.");
      },
      onError: (err) => {
        setFailedNode(nodeLog[nodeLog.length - 1] ?? "loader");
        setWhichStepFailed(1);
        setErrorInfo(classifyError(err));
        setStage("idle");
        setStatusMessage("");
      },
    });
  };

  /**
   * notesArg: the composed review string from the gap editor. Passed through
   * explicitly because setUserNotes in the modal doesn't commit synchronously
   * (stale-state race, live-run evidence Nov 5 2026). Retries and the
   * zero-gap auto-run fall back to the ref, which always holds the latest
   * typed context.
   */
  const handleOptimizeStep2 = (notesArg?: string) => {
    setErrorInfo(null);
    setWarningMessage(null);
    setNodeLog([]);
    setFailedNode(null);
    setWhichStepFailed(null);
    setStage("step2_running");
    setStatusMessage("Resuming pipeline from checkpoint...");
    setLastEventAt(Date.now());

    const fd = buildFormData();
    fd.append("notes", notesArg ?? userNotesRef.current);
    // BB1 Phase B (PB3): ask the server to echo the parsed original in the
    // result so the before/after diff also works for uploaded resumes.
    fd.append("include_diff", "original");

    runStep2(fd, {
      onStatus: handleStreamStatus,
      onWarning: (msg) => setWarningMessage(msg),
      onSession: (s) => setSession(s),
      onResult: (res) => {
        setSession(res.session);
        setResult(res);
        setStage("step2_done");
        setStatusMessage("Career documents generated successfully.");
        setShowForm(false);
      },
      onError: (err) => {
        setFailedNode(nodeLog[nodeLog.length - 1] ?? "improver");
        setWhichStepFailed(2);
        setErrorInfo(classifyError(err));
        // With a gap review: return to the checkpoint (reviewOpen is true →
        // the modal reopens, notes intact). In the straight-through run
        // there IS no checkpoint — stay on step2_running so the timeline
        // marks the failure at its actual node instead of a review that
        // never existed.
        if (reviewOpen) setStage("step1_review");
        setStatusMessage("");
      },
    });
  };

  // Retry: rerun whichever step failed (same payloads; session untouched)
  const handleRetry = () => {
    if (whichStepFailed === 1) handleAnalyzeStep1();
    else if (whichStepFailed === 2) handleOptimizeStep2();
  };

  const isProcessing = stage === "step1_running" || stage === "step2_running";

  return (
    <div className="min-h-[100dvh] bg-canvas text-ink flex flex-col font-sans selection:bg-accent-tint">
      {/* 1. Header */}
      <Header
        healthOk={healthOk}
        healthMessage={healthMessage}
        onReset={handleReset}
        onOpenSettings={() => setIsSettingsOpen(true)}
        isProcessing={isProcessing}
      />

      {/* 2. Main content */}
      <main className="flex-1 w-full px-3 sm:px-6 pt-5 pb-14">
        {/* Flow banner: gap analysis done, but step 2 is running's timeline handle */}
        {stage === "step1_review" && <p className="sr-only">Step one complete. Review and continue to step two.</p>}

        {/* Run failure card (classified, with action) */}
        {errorInfo && (
          <ErrorState
            error={errorInfo}
            onRetry={handleRetry}
            onOpenSettings={() => setIsSettingsOpen(true)}
            onDismiss={() => setErrorInfo(null)}
          />
        )}

        {/* Mid-stream warning banner (e.g., auto-detected corrections) */}
        {warningMessage && (
          <div
            role="status"
            className="mb-6 p-4 rounded-md2 bg-warn-tint border border-warn/30 text-2xs text-warn flex items-start justify-between gap-3 animate-in"
          >
            <span className="leading-relaxed">{warningMessage}</span>
            <button
              type="button"
              onClick={() => setWarningMessage(null)}
              aria-label="Dismiss warning"
              className="p-0.5 rounded-sm2 hover:text-warn transition-colors shrink-0"
            >
              <X weight="bold" className="w-3.5 h-3.5" />
            </button>
          </div>
        )}

        {/* Hero — compressed for panel width */}
        <section className="mb-5">
          <h1 className="text-lg font-semibold tracking-tight text-ink leading-snug max-w-2xl">
            Match your resume to the job posting
          </h1>
          <p className="text-xs text-ink-2 leading-relaxed mt-1 max-w-[65ch]">
            Scans gaps between the posting and your resume, then rewrites your documents on honest claims.
          </p>
        </section>

        {/* 3. Pipeline timeline */}
        <StepTimeline
          stage={stage}
          statusMessage={statusMessage}
          nodeLog={nodeLog}
          lastEventAt={lastEventAt}
          failedNode={failedNode}
        />

        {/* Skeleton while scanning */}
        {stage === "step1_running" && <ReviewSkeleton />}

        {/* 4. Inputs */}
        {stage === "step2_done" && !showForm && (
          <div className="panel p-3.5 mb-6 flex flex-col sm:flex-row sm:items-center justify-between gap-3 animate-in">
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-2xs min-w-0">
              <span className="flex items-center gap-1.5 text-ink-2 min-w-0">
                <FileText weight="bold" className="w-3.5 h-3.5 text-accent shrink-0" />
                <span className="truncate">
                  {jobMeta?.title && jobMeta?.company
                    ? `${jobMeta.title} · ${jobMeta.company}`
                    : jobMode === "url"
                    ? jobUrl || "URL provided"
                    : `${jobText.trim().split(/\s+/).filter(Boolean).length} word job description`}
                </span>
              </span>
              <span className="text-ink-3 font-mono">{resumeMode === "upload" && resumeFile ? resumeFile.name : `${resumeText.length} chars`}</span>
            </div>
            <button
              type="button"
              onClick={() => setShowForm(true)}
              className="btn btn-secondary btn-sm self-start sm:self-auto shrink-0"
            >
              Edit inputs & rerun
            </button>
          </div>
        )}
        {showForm && (
          <>
            <InputSection
              jobText={jobText}
              setJobText={setJobText}
              jobUrl={jobUrl}
              setJobUrl={setJobUrl}
              jobMode={jobMode}
              setJobMode={setJobMode}
              jobMeta={jobMeta}
              setJobMeta={setJobMeta}
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

            {/* Optional optimizer context. Feeds Step 2 even when the scan
                finds no gaps (straight-through run) and pre-fills the gap
                editor when a review checkpoint opens. Survives reloads. */}
            <div className="panel p-3.5 mt-3">
              <label htmlFor="optimizer-notes" className="text-2xs font-semibold text-ink">
                Add context for the optimizer <span className="font-normal text-ink-3">(optional)</span>
              </label>
              <p className="text-2xs text-ink-3 leading-relaxed mt-1">
                Projects or experience that prove a missing skill, tools to emphasize, or things to leave out.
                This rides along with every run until you erase it (Settings · Reset).
              </p>
              <textarea
                id="optimizer-notes"
                value={userNotes}
                onChange={(e) => setUserNotes(e.target.value)}
                rows={3}
                maxLength={3000}
                disabled={stage === "step1_running" || stage === "step2_running"}
                placeholder={`e.g. "I built an n8n workflow cutting delivery errors 30% — feature it." or "Do NOT feature MCP, I have no hands-on experience."`}
                className="input mt-2 w-full"
              />
            </div>
          </>
        )}

        {/* 5. Human-in-the-loop review checkpoint (skipped when the scan
             found no gaps — data.session flows straight to Step 2) */}
        {reviewOpen && (stage === "step1_review" || stage === "step2_running") && (
          <HumanReviewModal
            missingSkills={missingSkills}
            userNotes={userNotes}
            setUserNotes={setUserNotes}
            onSubmit={(notes) => handleOptimizeStep2(notes)}
            isProcessing={stage === "step2_running"}
          />
        )}

        {/* 6. Results (with skeleton while step 2 streams) */}
        {stage === "step2_running" && <ResultsSkeleton />}
        {result && (
          <div ref={resultsRef} className="scroll-mt-16">
            <ResultsDashboard
              result={result}
              originalResume={result.original_resume
                ?? (resumeMode === "paste" ? resumeText : undefined)}
            />
          </div>
        )}

        {/* BB1: application history (persistent profile store, auto-recorded) */}
        <HistoryPanel />
      </main>

      {/* Footer */}
      <footer className="w-full border-t border-line py-4 px-3 sm:px-6 text-2xs text-ink-3 text-center font-mono">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>AI Resume Optimizer · LangGraph pipeline</span>
          <span>Resume text stays in this browser</span>
        </div>
      </footer>

      {/* Toasts */}
      <ToastStack toasts={toasts} onDismiss={dismiss} />

      {/* Modals */}
      <Onboarding open={onboardingOpen} onDone={() => setOnboardingOpen(false)} />
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        onSettingsSaved={refreshHealth}
      />
    </div>
  );
}

export default App;
