import React, { useRef, useState } from "react";
import {
  FileText,
  Link as LinkIcon,
  UploadSimple,
  X,
  MagnifyingGlass,
  Briefcase,
  WarningCircle,
  CheckCircle,
} from "@phosphor-icons/react";

/**
 * Inputs panel. Two cards side by side (stacked at panel width): the target
 * job and the candidate resume. The detected-job card lives here so editing
 * happens where the data shows, not in a detached banner.
 */

export interface JobMeta {
  title: string;
  company: string;
}

interface InputSectionProps {
  jobText: string;
  setJobText: (val: string) => void;
  jobUrl: string;
  setJobUrl: (val: string) => void;
  jobMode: "text" | "url";
  setJobMode: (mode: "text" | "url") => void;
  /** truthy only when a scrape or page message delivered the job posting */
  jobMeta: JobMeta | null;
  setJobMeta: (meta: JobMeta | null) => void;
  resumeText: string;
  setResumeText: (val: string) => void;
  resumeFile: File | null;
  setResumeFile: (file: File | null) => void;
  resumeMode: "upload" | "paste";
  setResumeMode: (mode: "upload" | "paste") => void;
  onAnalyze: () => void;
  isProcessing: boolean;
  disabled: boolean;
}

/** Mirrors agent/config.py MAX_PDF_BYTES (5 MB) for the client-side gate. */
const MAX_FILE_MB = 5;

export const InputSection: React.FC<InputSectionProps> = ({
  jobText,
  setJobText,
  jobUrl,
  setJobUrl,
  jobMode,
  setJobMode,
  jobMeta,
  setJobMeta,
  resumeText,
  setResumeText,
  resumeFile,
  setResumeFile,
  resumeMode,
  setResumeMode,
  onAnalyze,
  isProcessing,
  disabled,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const [fileError, setFileError] = useState<string | null>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
    else if (e.type === "dragleave") setDragActive(false);
  };

  const acceptFile = (file: File): boolean => {
    const ext = file.name.split(".").pop()?.toLowerCase();
    if (ext !== "pdf" && ext !== "docx") {
      setFileError(`${file.name.slice(0, 60)} isn't a PDF or DOCX file.`);
      return false;
    }
    if (file.size > MAX_FILE_MB * 1024 * 1024) {
      setFileError(`${file.name.slice(0, 40)} is over the ${MAX_FILE_MB} MB limit.`);
      return false;
    }
    setFileError(null);
    setResumeFile(file);
    return true;
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    if (file) acceptFile(file);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) acceptFile(file);
    e.target.value = ""; // re-selecting the same file must still fire change
  };

  const wordCount = (text: string) => text.trim().split(/\s+/).filter(Boolean).length;

  const isJobReady =
    jobMode === "text"
      ? jobText.trim().length > 20
      : jobUrl.trim().startsWith("http");
  const isResumeReady =
    resumeMode === "upload" ? resumeFile !== null : resumeText.trim().length > 50;
  const canSubmit = isJobReady && isResumeReady && !disabled && !isProcessing;

  const readiness = [
    { ok: isJobReady, label: "Job description added" },
    { ok: isResumeReady, label: "Resume added" },
  ];

  return (
    <div className="w-full grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
      {/* 1. Target job card */}
      <section className="panel p-4 flex flex-col" aria-label="Job description input">
        {jobMeta && (
          <div className="mb-3 p-3 rounded-md2 bg-accent-tint border border-accent/25 animate-in">
            <div className="flex items-center justify-between gap-2 mb-2">
              <span className="flex items-center gap-1.5 text-2xs font-semibold text-accent uppercase tracking-wide">
                <Briefcase weight="bold" className="w-3.5 h-3.5" />
                Detected from page
              </span>
              <button
                type="button"
                onClick={() => setJobMeta(null)}
                aria-label="Clear detected job title and company"
                className="p-0.5 rounded-sm2 text-accent/70 hover:text-accent transition-colors"
              >
                <X weight="bold" className="w-3.5 h-3.5" />
              </button>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              <label className="block">
                <span className="sr-only">Job title</span>
                <input
                  type="text"
                  value={jobMeta.title}
                  onChange={(e) => setJobMeta({ ...jobMeta, title: e.target.value })}
                  placeholder="Job title"
                  aria-label="Edit detected job title"
                  className="input input-sm"
                />
              </label>
              <label className="block">
                <span className="sr-only">Company</span>
                <input
                  type="text"
                  value={jobMeta.company}
                  onChange={(e) => setJobMeta({ ...jobMeta, company: e.target.value })}
                  placeholder="Company"
                  aria-label="Edit detected company"
                  className="input input-sm"
                />
              </label>
            </div>
          </div>
        )}

        <div className="flex items-start justify-between gap-2 mb-3">
          <div>
            <h3 className="text-sm font-semibold text-ink">Target job</h3>
            <p className="text-2xs text-ink-3 mt-0.5">What you're applying for</p>
          </div>
          <div className="flex items-center p-0.5 rounded-md2 bg-inset border border-line" role="group" aria-label="Job input mode">
            <button
              type="button"
              onClick={() => setJobMode("text")}
              aria-pressed={jobMode === "text"}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-sm2 text-2xs font-medium transition-all ${
                jobMode === "text" ? "bg-raised text-ink shadow-card" : "text-ink-3 hover:text-ink"
              }`}
            >
              <FileText weight="bold" className="w-3 h-3" />
              <span>Paste text</span>
            </button>
            <button
              type="button"
              onClick={() => setJobMode("url")}
              aria-pressed={jobMode === "url"}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-sm2 text-2xs font-medium transition-all ${
                jobMode === "url" ? "bg-raised text-ink shadow-card" : "text-ink-3 hover:text-ink"
              }`}
            >
              <LinkIcon weight="bold" className="w-3 h-3" />
              <span>Job URL</span>
            </button>
          </div>
        </div>

        {jobMode === "text" ? (
          <div>
            <label htmlFor="job-description-textarea" className="field-label">
              Job responsibilities and qualifications
            </label>
            <textarea
              id="job-description-textarea"
              rows={9}
              value={jobText}
              onChange={(e) => setJobText(e.target.value)}
              placeholder="Paste the full job posting here (requirements, responsibilities, tech stack)..."
              disabled={disabled || isProcessing}
              className="input font-mono resize-none"
              aria-describedby="job-wordcount"
            />
            <div className="flex justify-between items-center mt-1.5 text-2xs text-ink-3 font-mono">
              <span id="job-wordcount">{jobText ? `${wordCount(jobText)} words` : "Empty"}</span>
              {jobText && (
                <button
                  type="button"
                  onClick={() => setJobText("")}
                  className="hover:text-ink transition-colors px-1"
                >
                  Clear
                </button>
              )}
            </div>
          </div>
        ) : (
          <div className="py-2">
            <label htmlFor="job-url-input" className="field-label">
              Public job posting URL
            </label>
            <input
              id="job-url-input"
              type="url"
              inputMode="url"
              value={jobUrl}
              onChange={(e) => setJobUrl(e.target.value)}
              placeholder="https://boards.greenhouse.io/company/jobs/12345"
              disabled={disabled || isProcessing}
              className="input font-mono"
            />
            <p className="text-2xs text-ink-3 mt-2 leading-relaxed">
              Works with LinkedIn, Greenhouse, Lever, Workday, and Ashby postings.
            </p>
          </div>
        )}
      </section>

      {/* 2. Resume card */}
      <section className="panel p-4 flex flex-col" aria-label="Resume input">
        <div className="flex items-start justify-between gap-2 mb-3">
          <div>
            <h3 className="text-sm font-semibold text-ink">Your resume</h3>
            <p className="text-2xs text-ink-3 mt-0.5">The honest baseline</p>
          </div>
          <div className="flex items-center p-0.5 rounded-md2 bg-inset border border-line" role="group" aria-label="Resume input mode">
            <button
              type="button"
              onClick={() => setResumeMode("upload")}
              aria-pressed={resumeMode === "upload"}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-sm2 text-2xs font-medium transition-all ${
                resumeMode === "upload" ? "bg-raised text-ink shadow-card" : "text-ink-3 hover:text-ink"
              }`}
            >
              <UploadSimple weight="bold" className="w-3 h-3" />
              <span>Upload</span>
            </button>
            <button
              type="button"
              onClick={() => setResumeMode("paste")}
              aria-pressed={resumeMode === "paste"}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-sm2 text-2xs font-medium transition-all ${
                resumeMode === "paste" ? "bg-raised text-ink shadow-card" : "text-ink-3 hover:text-ink"
              }`}
            >
              <FileText weight="bold" className="w-3 h-3" />
              <span>Paste text</span>
            </button>
          </div>
        </div>

        {resumeMode === "upload" ? (
          <div>
            <label className="field-label">Resume document — PDF or DOCX, up to {MAX_FILE_MB} MB</label>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
              onChange={handleFileChange}
              className="sr-only"
              aria-label="Resume file"
            />
            {resumeFile ? (
              <div className="p-3 rounded-md2 bg-inset border border-accent/30 flex items-center justify-between gap-2">
                <div className="flex items-center gap-2.5 min-w-0">
                  <div className="w-9 h-9 rounded-sm2 bg-accent-tint border border-accent/20 flex items-center justify-center text-accent shrink-0">
                    <FileText weight="bold" className="w-4 h-4" />
                  </div>
                  <div className="min-w-0">
                    <p className="text-xs font-medium text-ink truncate">{resumeFile.name}</p>
                    <p className="text-2xs text-ink-3 font-mono mt-0.5">
                      {(resumeFile.size / 1024).toFixed(0)} KB
                    </p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    setResumeFile(null);
                    setFileError(null);
                  }}
                  aria-label={`Remove file ${resumeFile.name}`}
                  disabled={disabled || isProcessing}
                  className="p-1.5 rounded-sm2 text-ink-3 hover:text-danger hover:bg-danger-tint transition-colors disabled:opacity-40"
                >
                  <X weight="bold" className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                disabled={disabled || isProcessing}
                aria-describedby={fileError ? "resume-file-error" : "resume-upload-hint"}
                className={`w-full border-2 border-dashed rounded-md2 px-4 py-8 flex flex-col items-center justify-center text-center cursor-pointer transition-all disabled:opacity-50 disabled:cursor-not-allowed ${
                  dragActive
                    ? "border-accent bg-accent-tint"
                    : "border-line hover:border-line-strong bg-inset"
                }`}
              >
                <div className="w-10 h-10 rounded-md2 bg-raised border border-line flex items-center justify-center text-accent mb-2.5">
                  <UploadSimple weight="bold" className="w-4.5 h-4.5" />
                </div>
                <p className="text-xs font-medium text-ink">Click to browse, or drop the file here</p>
                <p id="resume-upload-hint" className="text-2xs text-ink-3 mt-1 font-mono">
                  PDF or DOCX, max {MAX_FILE_MB} MB
                </p>
              </button>
            )}
            {fileError && (
              <p id="resume-file-error" role="alert" className="flex items-start gap-1.5 mt-2 text-2xs text-danger leading-relaxed">
                <WarningCircle weight="bold" className="w-3.5 h-3.5 shrink-0 mt-px" />
                {fileError}
              </p>
            )}
          </div>
        ) : (
          <div>
            <label htmlFor="resume-markdown-textarea" className="field-label">
              Resume content (markdown or plain text)
            </label>
            <textarea
              id="resume-markdown-textarea"
              rows={9}
              value={resumeText}
              onChange={(e) => setResumeText(e.target.value)}
              placeholder="Paste your resume here (sections, bullets, skills, education)..."
              disabled={disabled || isProcessing}
              className="input font-mono resize-none"
              aria-describedby="resume-wordcount"
            />
            <div className="flex justify-between items-center mt-1.5 text-2xs text-ink-3 font-mono">
              <span id="resume-wordcount">{resumeText ? `${wordCount(resumeText)} words` : "Empty"}</span>
              {resumeText && (
                <button
                  type="button"
                  onClick={() => setResumeText("")}
                  className="hover:text-ink transition-colors px-1"
                >
                  Clear
                </button>
              )}
            </div>
          </div>
        )}

        {/* Readiness + action: the checklist doubles as the why-disabled explanation */}
        <div className="pt-3 mt-auto border-t border-line">
          <ul className="flex items-center gap-3 mb-3" aria-label="Ready to run">
            {readiness.map((r) => (
              <li key={r.label} className="flex items-center gap-1 text-2xs">
                <CheckCircle
                  weight="bold"
                  className={`w-3 h-3 ${r.ok ? "text-ok" : "text-ink-3/60"}`}
                  aria-hidden="true"
                />
                <span className={r.ok ? "text-ink-2" : "text-ink-3"}>{r.label}</span>
              </li>
            ))}
          </ul>
          <button type="button" onClick={onAnalyze} disabled={!canSubmit} className="btn btn-primary w-full">
            {isProcessing ? (
              <>
                <span className="w-3.5 h-3.5 border-2 border-accent-contrast border-t-transparent rounded-full animate-spin-slow" />
                <span>Scanning gaps...</span>
              </>
            ) : (
              <>
                <MagnifyingGlass weight="bold" className="w-4 h-4" />
                <span>Step 1 — Analyze skill gaps</span>
              </>
            )}
          </button>
        </div>
      </section>
    </div>
  );
};
