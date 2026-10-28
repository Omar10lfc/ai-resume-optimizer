import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  DownloadSimple,
  Copy,
  Check,
  FilePdf,
  FileDoc,
  FileCode,
  Article,
  ChartBar,
  Question,
  Sparkle,
} from "@phosphor-icons/react";
import type { OptimizationResult, Artifact } from "../types";

interface ResultsDashboardProps {
  result: OptimizationResult;
}

export const ResultsDashboard: React.FC<ResultsDashboardProps> = ({ result }) => {
  const [activeTab, setActiveTab] = useState<"resume" | "cover" | "ats" | "interview">("resume");
  const [copied, setCopied] = useState<string | null>(null);

  const copyToClipboard = (text: string, label: string) => {
    navigator.clipboard.writeText(text);
    setCopied(label);
    setTimeout(() => setCopied(null), 2000);
  };

  // Score-class taxonomy comes from the server (good/warn/bad thresholds in
  // one place). The numeric badge still shows the composite score itself.
  const getScoreBadge = (scoreClass: "good" | "warn" | "bad") => {
    if (scoreClass === "good") {
      return {
        bg: "bg-emerald-950/40 border-emerald-500/40 text-emerald-400",
        label: "Ready to Apply",
        statusClass: "text-emerald-400",
      };
    }
    if (scoreClass === "warn") {
      return {
        bg: "bg-amber-950/40 border-amber-500/40 text-amber-400",
        label: "Moderate Match",
        statusClass: "text-amber-400",
      };
    }
    return {
      bg: "bg-rose-950/40 border-rose-500/40 text-rose-400",
      label: "Needs Review",
      statusClass: "text-rose-400",
    };
  };

  const badge = getScoreBadge(result.score_class);

  const getArtifactIcon = (kind: Artifact["kind"]) => {
    switch (kind) {
      case "resume-pdf":
      case "cover-pdf":
        return <FilePdf weight="bold" className="w-4 h-4 text-rose-400" />;
      case "resume-docx":
        return <FileDoc weight="bold" className="w-4 h-4 text-blue-400" />;
      case "resume-tex":
        return <FileCode weight="bold" className="w-4 h-4 text-emerald-400" />;
    }
  };

  const getArtifactLabel = (kind: Artifact["kind"]) => {
    switch (kind) {
      case "resume-pdf":
        return "Resume PDF (LaTeX)";
      case "cover-pdf":
        return "Cover Letter PDF";
      case "resume-docx":
        return "Resume DOCX (Word)";
      case "resume-tex":
        return "LaTeX Source (.tex)";
    }
  };

  const markdownComponents = {
    h1: ({ ...props }) => (
      <h1 className="text-lg font-bold text-[#f8fafc] mt-4 mb-3 pb-1 border-b border-[#1e2436]" {...props} />
    ),
    h2: ({ ...props }) => (
      <h2 className="text-base font-semibold text-[#f8fafc] mt-5 mb-2 pb-1 border-b border-[#1e2436] flex items-center gap-2" {...props} />
    ),
    h3: ({ ...props }) => (
      <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-[#06b6d4] mt-4 mb-2" {...props} />
    ),
    p: ({ ...props }) => (
      <p className="text-sm text-[#cbd5e1] leading-relaxed my-2" {...props} />
    ),
    strong: ({ ...props }) => (
      <strong className="font-semibold text-[#f8fafc]" {...props} />
    ),
    blockquote: ({ ...props }) => (
      <blockquote className="border-l-2 border-[#06b6d4]/70 bg-[#151926]/60 pl-3.5 py-2 my-2.5 rounded-r-lg text-xs text-[#94a3b8] italic" {...props} />
    ),
    ul: ({ ...props }) => (
      <ul className="list-disc pl-5 my-2.5 space-y-1.5 text-sm text-[#cbd5e1]" {...props} />
    ),
    ol: ({ ...props }) => (
      <ol className="list-decimal pl-5 my-2.5 space-y-1.5 text-sm text-[#cbd5e1]" {...props} />
    ),
    li: ({ ...props }) => (
      <li className="text-sm text-[#cbd5e1] leading-relaxed" {...props} />
    ),
    code: ({ ...props }) => (
      <code className="font-mono text-xs px-1.5 py-0.5 rounded bg-[#151926] text-[#06b6d4] border border-[#1e2436]" {...props} />
    ),
  };

  return (
    <div className="w-full bg-[#10131c] border border-[#1e2436] rounded-2xl p-6 mb-12 shadow-xl">
      {/* 1. Score & Overview Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 pb-6 mb-6 border-b border-[#1e2436]">
        <div className="flex items-center gap-5">
          {/* Score Dial */}
          <div className="w-20 h-20 rounded-2xl bg-[#151926] border border-[#1e2436] flex flex-col items-center justify-center p-2 text-center shrink-0 shadow-inner">
            <span className={`text-2xl font-bold font-mono tracking-tight ${badge.statusClass}`}>
              {result.score}
            </span>
            <span className="text-[10px] text-[#64748b] font-mono uppercase tracking-wider mt-0.5">/ 100</span>
          </div>

          <div>
            <div className="flex items-center gap-2">
              <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold border ${badge.bg}`}>
                {badge.label}
              </span>
              <span className="text-xs text-[#94a3b8] font-mono">Composite Quality</span>
            </div>
            <h3 className="text-lg font-semibold text-[#f8fafc] mt-1.5">
              Optimized Career Documents Generated
            </h3>
            <p className="text-xs text-[#64748b] mt-0.5">
              60% deterministic ATS keywords ({result.ats_percentage}%) + 40% presentation quality rubric
            </p>
          </div>
        </div>

        {/* 2. Download Center */}
        <div className="flex flex-wrap items-center gap-2">
          {result.artifacts.map((art) => (
            <a
              key={art.url}
              href={`/${art.url}`}
              download={art.filename}
              className="flex items-center gap-2 px-3.5 py-2 rounded-xl bg-[#151926] hover:bg-[#1b2030] border border-[#1e2436] hover:border-[#2a334c] text-xs font-medium text-[#f8fafc] transition-all shadow-sm active:scale-[0.98]"
            >
              {getArtifactIcon(art.kind)}
              <span>{getArtifactLabel(art.kind)}</span>
              <DownloadSimple weight="bold" className="w-3.5 h-3.5 text-[#94a3b8]" />
            </a>
          ))}
        </div>
      </div>

      {/* 3. Tab Bar */}
      <div className="flex items-center gap-1 p-1 rounded-xl bg-[#090a0f] border border-[#1e2436] mb-6 overflow-x-auto">
        <button
          type="button"
          onClick={() => setActiveTab("resume")}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all shrink-0 ${
            activeTab === "resume"
              ? "bg-[#151926] text-[#06b6d4] shadow-sm border border-[#1e2436]"
              : "text-[#94a3b8] hover:text-[#f8fafc]"
          }`}
        >
          <Article weight="bold" className="w-4 h-4" />
          <span>Optimized Resume</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab("cover")}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all shrink-0 ${
            activeTab === "cover"
              ? "bg-[#151926] text-[#06b6d4] shadow-sm border border-[#1e2436]"
              : "text-[#94a3b8] hover:text-[#f8fafc]"
          }`}
        >
          <Sparkle weight="bold" className="w-4 h-4" />
          <span>Tailored Cover Letter</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab("ats")}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all shrink-0 ${
            activeTab === "ats"
              ? "bg-[#151926] text-[#06b6d4] shadow-sm border border-[#1e2436]"
              : "text-[#94a3b8] hover:text-[#f8fafc]"
          }`}
        >
          <ChartBar weight="bold" className="w-4 h-4" />
          <span>ATS Keyword Breakdown</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab("interview")}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all shrink-0 ${
            activeTab === "interview"
              ? "bg-[#151926] text-[#06b6d4] shadow-sm border border-[#1e2436]"
              : "text-[#94a3b8] hover:text-[#f8fafc]"
          }`}
        >
          <Question weight="bold" className="w-4 h-4" />
          <span>Interview Prep and Questions</span>
        </button>
      </div>

      {/* 4. Tab Content Pane */}
      <div className="relative">
        {activeTab === "resume" && (
          <div className="bg-[#090a0f] border border-[#1e2436] rounded-xl p-6">
            <div className="flex justify-between items-center pb-4 mb-4 border-b border-[#1e2436]">
              <span className="text-xs font-mono text-[#94a3b8]">Single-page formatted resume preview</span>
              <button
                type="button"
                onClick={() => copyToClipboard(result.optimized_resume, "resume")}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#151926] border border-[#1e2436] text-xs font-medium text-[#f8fafc] hover:bg-[#1b2030] transition-all"
              >
                {copied === "resume" ? (
                  <>
                    <Check weight="bold" className="w-3.5 h-3.5 text-emerald-400" />
                    <span className="text-emerald-400">Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy weight="bold" className="w-3.5 h-3.5" />
                    <span>Copy Markdown</span>
                  </>
                )}
              </button>
            </div>
            <div className="max-w-none text-sm text-[#cbd5e1] font-mono leading-relaxed overflow-x-auto">
              <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                {result.optimized_resume}
              </ReactMarkdown>
            </div>
          </div>
        )}

        {activeTab === "cover" && (
          <div className="bg-[#090a0f] border border-[#1e2436] rounded-xl p-6">
            <div className="flex justify-between items-center pb-4 mb-4 border-b border-[#1e2436]">
              <span className="text-xs font-mono text-[#94a3b8]">Prose cover letter with role hooks</span>
              <button
                type="button"
                onClick={() => copyToClipboard(result.cover_letter, "cover")}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#151926] border border-[#1e2436] text-xs font-medium text-[#f8fafc] hover:bg-[#1b2030] transition-all"
              >
                {copied === "cover" ? (
                  <>
                    <Check weight="bold" className="w-3.5 h-3.5 text-emerald-400" />
                    <span className="text-emerald-400">Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy weight="bold" className="w-3.5 h-3.5" />
                    <span>Copy Letter</span>
                  </>
                )}
              </button>
            </div>
            <div className="max-w-none text-sm text-[#cbd5e1] leading-relaxed">
              <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                {result.cover_letter}
              </ReactMarkdown>
            </div>
          </div>
        )}

        {activeTab === "ats" && (
          <div className="bg-[#090a0f] border border-[#1e2436] rounded-xl p-6">
            <div className="flex justify-between items-center pb-4 mb-4 border-b border-[#1e2436]">
              <span className="text-xs font-mono text-[#94a3b8]">Tier-weighted keyword match report</span>
              <button
                type="button"
                onClick={() => copyToClipboard(result.ats_result, "ats")}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#151926] border border-[#1e2436] text-xs font-medium text-[#f8fafc] hover:bg-[#1b2030] transition-all"
              >
                {copied === "ats" ? (
                  <>
                    <Check weight="bold" className="w-3.5 h-3.5 text-emerald-400" />
                    <span className="text-emerald-400">Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy weight="bold" className="w-3.5 h-3.5" />
                    <span>Copy ATS Report</span>
                  </>
                )}
              </button>
            </div>
            <div className="max-w-none text-sm text-[#cbd5e1] font-mono leading-relaxed overflow-x-auto whitespace-pre-wrap">
              <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                {result.ats_result}
              </ReactMarkdown>
            </div>
          </div>
        )}

        {activeTab === "interview" && (
          <div className="bg-[#090a0f] border border-[#1e2436] rounded-xl p-6">
            <div className="flex justify-between items-center pb-4 mb-4 border-b border-[#1e2436]">
              <span className="text-xs font-mono text-[#94a3b8]">Targeted role questions and STAR framework stories</span>
              <button
                type="button"
                onClick={() => copyToClipboard(result.interview_questions, "interview")}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#151926] border border-[#1e2436] text-xs font-medium text-[#f8fafc] hover:bg-[#1b2030] transition-all"
              >
                {copied === "interview" ? (
                  <>
                    <Check weight="bold" className="w-3.5 h-3.5 text-emerald-400" />
                    <span className="text-emerald-400">Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy weight="bold" className="w-3.5 h-3.5" />
                    <span>Copy Questions</span>
                  </>
                )}
              </button>
            </div>
            <div className="max-w-none text-sm text-[#cbd5e1] leading-relaxed">
              <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                {result.interview_questions}
              </ReactMarkdown>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
