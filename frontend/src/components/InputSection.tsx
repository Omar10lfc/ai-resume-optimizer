import React, { useState, useRef } from "react";
import {
  FileText,
  Link as LinkIcon,
  UploadSimple,
  X,
  MagnifyingGlass,
} from "@phosphor-icons/react";

interface InputSectionProps {
  jobText: string;
  setJobText: (val: string) => void;
  jobUrl: string;
  setJobUrl: (val: string) => void;
  jobMode: "text" | "url";
  setJobMode: (mode: "text" | "url") => void;
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

export const InputSection: React.FC<InputSectionProps> = ({
  jobText,
  setJobText,
  jobUrl,
  setJobUrl,
  jobMode,
  setJobMode,
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

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      const ext = file.name.split(".").pop()?.toLowerCase();
      if (ext === "pdf" || ext === "docx") {
        setResumeFile(file);
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setResumeFile(e.target.files[0]);
    }
  };

  const isJobReady = (jobMode === "text" && jobText.trim().length > 20) || (jobMode === "url" && jobUrl.trim().startsWith("http"));
  const isResumeReady = (resumeMode === "upload" && resumeFile !== null) || (resumeMode === "paste" && resumeText.trim().length > 50);
  const canSubmit = isJobReady && isResumeReady && !disabled && !isProcessing;

  return (
    <div className="w-full grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
      {/* 1. Job Description Panel */}
      <div className="bg-[#10131c] border border-[#1e2436] rounded-2xl p-5 flex flex-col justify-between">
        <div>
          <div className="flex items-center justify-between mb-4">
            <div>
              <span className="text-xs font-mono uppercase tracking-wider text-[#06b6d4]">Target Position</span>
              <h3 className="text-base font-semibold text-[#f8fafc] mt-0.5">Job Description</h3>
            </div>

            {/* Mode toggle */}
            <div className="flex items-center p-1 rounded-xl bg-[#090a0f] border border-[#1e2436]">
              <button
                type="button"
                onClick={() => setJobMode("text")}
                className={`flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-medium transition-all ${
                  jobMode === "text"
                    ? "bg-[#151926] text-[#f8fafc] shadow-sm"
                    : "text-[#94a3b8] hover:text-[#f8fafc]"
                }`}
              >
                <FileText weight="bold" className="w-3.5 h-3.5" />
                <span>Paste Text</span>
              </button>
              <button
                type="button"
                onClick={() => setJobMode("url")}
                className={`flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-medium transition-all ${
                  jobMode === "url"
                    ? "bg-[#151926] text-[#f8fafc] shadow-sm"
                    : "text-[#94a3b8] hover:text-[#f8fafc]"
                }`}
              >
                <LinkIcon weight="bold" className="w-3.5 h-3.5" />
                <span>Job URL</span>
              </button>
            </div>
          </div>

          {jobMode === "text" ? (
            <div>
              <label htmlFor="job-description-textarea" className="block text-xs font-medium text-[#94a3b8] mb-2">
                Job Responsibilities and Qualifications
              </label>
              <textarea
                id="job-description-textarea"
                rows={9}
                value={jobText}
                onChange={(e) => setJobText(e.target.value)}
                placeholder="Paste the full job posting text here (requirements, responsibilities, tech stack)..."
                disabled={disabled || isProcessing}
                className="w-full p-3.5 rounded-xl bg-[#090a0f] border border-[#1e2436] text-sm text-[#f8fafc] placeholder-[#64748b] focus:outline-none focus:border-[#06b6d4]/60 transition-colors resize-none font-mono"
              />
              <div className="flex justify-between items-center mt-2 text-[11px] text-[#64748b] font-mono">
                <span>{jobText ? `${jobText.trim().split(/\s+/).length} words` : "Empty"}</span>
                {jobText && (
                  <button
                    type="button"
                    onClick={() => setJobText("")}
                    className="hover:text-[#f8fafc] transition-colors"
                  >
                    Clear
                  </button>
                )}
              </div>
            </div>
          ) : (
            <div className="py-8">
              <label htmlFor="job-url-input" className="block text-xs font-medium text-[#94a3b8] mb-2">
                Public Job Posting URL
              </label>
              <input
                id="job-url-input"
                type="url"
                value={jobUrl}
                onChange={(e) => setJobUrl(e.target.value)}
                placeholder="https://boards.greenhouse.io/company/jobs/12345"
                disabled={disabled || isProcessing}
                className="w-full p-3.5 rounded-xl bg-[#090a0f] border border-[#1e2436] text-sm text-[#f8fafc] placeholder-[#64748b] focus:outline-none focus:border-[#06b6d4]/60 transition-colors font-mono"
              />
              <p className="text-xs text-[#64748b] mt-2.5 leading-relaxed">
                Supports LinkedIn, Greenhouse, Lever, Workday, and standard corporate careers pages.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* 2. Resume Profile Panel */}
      <div className="bg-[#10131c] border border-[#1e2436] rounded-2xl p-5 flex flex-col justify-between">
        <div>
          <div className="flex items-center justify-between mb-4">
            <div>
              <span className="text-xs font-mono uppercase tracking-wider text-[#06b6d4]">Candidate Data</span>
              <h3 className="text-base font-semibold text-[#f8fafc] mt-0.5">Original Resume</h3>
            </div>

            {/* Mode toggle */}
            <div className="flex items-center p-1 rounded-xl bg-[#090a0f] border border-[#1e2436]">
              <button
                type="button"
                onClick={() => setResumeMode("upload")}
                className={`flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-medium transition-all ${
                  resumeMode === "upload"
                    ? "bg-[#151926] text-[#f8fafc] shadow-sm"
                    : "text-[#94a3b8] hover:text-[#f8fafc]"
                }`}
              >
                <UploadSimple weight="bold" className="w-3.5 h-3.5" />
                <span>Upload PDF / DOCX</span>
              </button>
              <button
                type="button"
                onClick={() => setResumeMode("paste")}
                className={`flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-medium transition-all ${
                  resumeMode === "paste"
                    ? "bg-[#151926] text-[#f8fafc] shadow-sm"
                    : "text-[#94a3b8] hover:text-[#f8fafc]"
                }`}
              >
                <FileText weight="bold" className="w-3.5 h-3.5" />
                <span>Paste Text</span>
              </button>
            </div>
          </div>

          {resumeMode === "upload" ? (
            <div>
              <label className="block text-xs font-medium text-[#94a3b8] mb-2">
                Resume Document (PDF or DOCX, max 5 MB)
              </label>

              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                onChange={handleFileChange}
                className="hidden"
              />

              {resumeFile ? (
                <div className="p-4 rounded-xl bg-[#090a0f] border border-[#06b6d4]/40 flex items-center justify-between">
                  <div className="flex items-center gap-3 truncate">
                    <div className="w-10 h-10 rounded-lg bg-[#06b6d4]/10 border border-[#06b6d4]/20 flex items-center justify-center text-[#06b6d4] shrink-0">
                      <FileText weight="bold" className="w-5 h-5" />
                    </div>
                    <div className="truncate">
                      <p className="text-sm font-medium text-[#f8fafc] truncate">{resumeFile.name}</p>
                      <p className="text-xs text-[#64748b] font-mono mt-0.5">
                        {(resumeFile.size / 1024).toFixed(1)} KB
                      </p>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => setResumeFile(null)}
                    disabled={disabled || isProcessing}
                    className="p-1.5 rounded-lg text-[#94a3b8] hover:text-rose-400 hover:bg-[#151926] transition-colors"
                  >
                    <X weight="bold" className="w-4 h-4" />
                  </button>
                </div>
              ) : (
                <div
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all flex flex-col items-center justify-center ${
                    dragActive
                      ? "border-[#06b6d4] bg-[#06b6d4]/5"
                      : "border-[#1e2436] hover:border-[#2a334c] bg-[#090a0f]"
                  }`}
                >
                  <div className="w-11 h-11 rounded-xl bg-[#151926] border border-[#1e2436] flex items-center justify-center text-[#06b6d4] mb-3">
                    <UploadSimple weight="bold" className="w-5 h-5" />
                  </div>
                  <p className="text-sm font-medium text-[#f8fafc]">Click to browse or drop resume file</p>
                  <p className="text-xs text-[#64748b] mt-1">Accepts PDF or Word DOCX files</p>
                </div>
              )}
            </div>
          ) : (
            <div>
              <label htmlFor="resume-markdown-textarea" className="block text-xs font-medium text-[#94a3b8] mb-2">
                Resume Content (Markdown or Raw Text)
              </label>
              <textarea
                id="resume-markdown-textarea"
                rows={9}
                value={resumeText}
                onChange={(e) => setResumeText(e.target.value)}
                placeholder="Paste your existing resume text (sections, bullets, skills, education)..."
                disabled={disabled || isProcessing}
                className="w-full p-3.5 rounded-xl bg-[#090a0f] border border-[#1e2436] text-sm text-[#f8fafc] placeholder-[#64748b] focus:outline-none focus:border-[#06b6d4]/60 transition-colors resize-none font-mono"
              />
              <div className="flex justify-between items-center mt-2 text-[11px] text-[#64748b] font-mono">
                <span>{resumeText ? `${resumeText.trim().split(/\s+/).length} words` : "Empty"}</span>
                {resumeText && (
                  <button
                    type="button"
                    onClick={() => setResumeText("")}
                    className="hover:text-[#f8fafc] transition-colors"
                  >
                    Clear
                  </button>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Action button */}
        <div className="pt-5 mt-5 border-t border-[#1e2436] flex justify-end">
          <button
            type="button"
            onClick={onAnalyze}
            disabled={!canSubmit}
            className="w-full sm:w-auto px-6 py-2.5 rounded-xl bg-[#06b6d4] hover:bg-[#0891b2] text-[#090a0f] font-semibold text-sm transition-all flex items-center justify-center gap-2 shadow-[0_0_20px_rgba(6,182,212,0.2)] disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none active:scale-[0.98]"
          >
            {isProcessing ? (
              <>
                <span className="w-4 h-4 border-2 border-[#090a0f] border-t-transparent rounded-full animate-spin" />
                <span>Running Gap Analysis...</span>
              </>
            ) : (
              <>
                <MagnifyingGlass weight="bold" className="w-4 h-4" />
                <span>Step 1: Analyze Skill Gaps</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
