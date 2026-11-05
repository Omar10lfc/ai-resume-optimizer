import React, { useCallback, useEffect, useRef, useState } from "react";
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
  Lightbulb,
  ArrowsLeftRight,
} from "@phosphor-icons/react";
import type { OptimizationResult } from "../types";
import { resolveArtifactUrl } from "../api";
import { ScoreGauge } from "./ScoreGauge";
import { DiffView } from "./DiffView";
import { Markdown } from "./Markdown";

type TabId = "resume" | "cover" | "ats" | "interview";

interface ResultsDashboardProps {
  result: OptimizationResult;
  /** Present only in paste mode: powers the before/after diff view. */
  originalResume?: string;
}

const TABS: Array<{ id: TabId; label: string; icon: React.ElementType }> = [
  { id: "resume", label: "Resume", icon: Article },
  { id: "cover", label: "Cover letter", icon: Sparkle },
  { id: "ats", label: "ATS report", icon: ChartBar },
  { id: "interview", label: "Interview prep", icon: Question },
];

const ARTIFACT_META: Record<string, { label: string; icon: React.ElementType; className: string }> = {
  "resume-pdf": { label: "Resume PDF", icon: FilePdf, className: "text-danger" },
  "cover-pdf": { label: "Cover letter PDF", icon: FilePdf, className: "text-danger" },
  "resume-docx": { label: "Resume DOCX", icon: FileDoc, className: "text-info" },
  "resume-tex": { label: "LaTeX source", icon: FileCode, className: "text-ok" },
};

function useCopyFeedback(): { copied: string | null; copy: (text: string, label: string) => void } {
  const [copied, setCopied] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);
  const copy = useCallback((text: string, label: string) => {
    navigator.clipboard.writeText(text).then(
      () => {
        setCopied(label);
        if (timer.current) clearTimeout(timer.current);
        timer.current = setTimeout(() => setCopied(null), 2000);
      },
      () => {
        // clipboard may be blocked in some embedded contexts; silent failure
        // is acceptable here since the text is still visible on screen
      }
    );
  }, []);
  return { copied, copy };
}

export const ResultsDashboard: React.FC<ResultsDashboardProps> = ({ result, originalResume }) => {
  const [activeTab, setActiveTab] = useState<TabId>("resume");
  const [artifactUrls, setArtifactUrls] = useState<Record<string, string>>({});
  const [showDiff, setShowDiff] = useState(false);
  const { copied, copy } = useCopyFeedback();
  const tabRefs = useRef(new Map<TabId, HTMLButtonElement | null>());

  // Artifact URLs resolve against the configured backend base. Resolution is
  // a string build (no network), so links may point at the raw relative path
  // immediately and are replaced in place — no skeletons needed, no shift.
  useEffect(() => {
    let isMounted = true;
    async function loadUrls() {
      const map: Record<string, string> = {};
      for (const art of result.artifacts) {
        map[art.url] = await resolveArtifactUrl(art.url);
      }
      if (isMounted) setArtifactUrls(map);
    }
    loadUrls();
    return () => {
      isMounted = false;
    };
  }, [result]);

  // ARIA tablist: roving tabindex with arrow keys, Home/End support.
  const onTablistKeyDown = (e: React.KeyboardEvent) => {
    const idx = TABS.findIndex((t) => t.id === activeTab);
    const move = (dir: number) => {
      e.preventDefault();
      const next = TABS[(idx + dir + TABS.length) % TABS.length];
      setActiveTab(next.id);
      tabRefs.current.get(next.id)?.focus();
    };
    if (e.key === "ArrowRight") move(1);
    else if (e.key === "ArrowLeft") move(-1);
    else if (e.key === "Home") {
      e.preventDefault();
      setActiveTab(TABS[0].id);
      tabRefs.current.get(TABS[0].id)?.focus();
    } else if (e.key === "End") {
      e.preventDefault();
      const last = TABS[TABS.length - 1];
      setActiveTab(last.id);
      tabRefs.current.get(last.id)?.focus();
    }
  };

  const canDiff = Boolean(originalResume && originalResume.trim().length > 40);

  const renderCopyRow = (label: string, text: string) => (
    <button type="button" onClick={() => copy(text, label)} className="btn btn-secondary btn-sm">
      {copied === label ? (
        <>
          <Check weight="bold" className="w-3.5 h-3.5 text-ok" />
          <span className="text-ok">Copied</span>
        </>
      ) : (
        <>
          <Copy weight="bold" className="w-3.5 h-3.5" />
          <span>Copy</span>
        </>
      )}
    </button>
  );

  return (
    <div className="panel p-4 sm:p-5 mb-8">
      {/* 1. Score head */}
      <div className="pb-4 mb-4 border-b border-line">
        <div className="flex items-center justify-between gap-3 mb-3">
          <h2 className="text-sm font-semibold text-ink">Your optimized documents</h2>
          <span className="text-2xs text-ink-3 font-mono truncate max-w-[45%]" title={result.session}>
            run {result.session ? result.session.slice(0, 8) : "—"}
          </span>
        </div>
        <ScoreGauge
          score={result.score}
          scoreClass={result.score_class}
          atsPercentage={result.ats_percentage}
          weights={result.weights}
        />
      </div>

      {/* 2. Download rows */}
      <div className="mb-4" aria-label="Downloads">
        <h3 className="field-label !mb-2">Downloads</h3>
        <div className="space-y-1.5">
          {result.artifacts.length === 0 && (
            <p className="px-3 py-2.5 rounded-md2 bg-warn-tint border border-warn/25 text-2xs text-warn" role="status">
              No files came back with this run. The text tabs below hold everything that was generated.
            </p>
          )}
          {result.artifacts.map((art) => {
            const meta = ARTIFACT_META[art.kind] ?? { label: art.kind, icon: FileCode, className: "text-ink" };
            const Icon = meta.icon;
            const href = artifactUrls[art.url] || `/${art.url}`;
            return (
              <a
                key={art.url}
                href={href}
                download={art.filename}
                className="flex items-center gap-2.5 px-3 py-2 rounded-md2 bg-raised hover:bg-raised-hover border border-line hover:border-line-strong transition-colors group min-h-[2.5rem]"
              >
                <Icon weight="bold" className={`w-4 h-4 shrink-0 ${meta.className}`} />
                <span className="flex-1 min-w-0">
                  <span className="block text-xs text-ink font-medium truncate">{meta.label}</span>
                  <span className="block text-2xs text-ink-3 font-mono truncate">{art.filename}</span>
                </span>
                <DownloadSimple weight="bold" className="w-4 h-4 text-ink-3 group-hover:text-accent transition-colors shrink-0" />
              </a>
            );
          })}
        </div>
      </div>

      {/* 3. Tabs */}
      <div
        role="tablist"
        aria-label="Result sections"
        onKeyDown={onTablistKeyDown}
        className="flex items-center gap-1 p-1 rounded-md2 bg-inset border border-line mb-3 overflow-x-auto"
      >
        {TABS.map((tab) => (
          <button
            key={tab.id}
            ref={(el) => {
              tabRefs.current.set(tab.id, el);
            }}
            id={`tab-${tab.id}`}
            role="tab"
            aria-selected={activeTab === tab.id}
            aria-controls={`panel-${tab.id}`}
            tabIndex={activeTab === tab.id ? 0 : -1}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-sm2 text-2xs font-semibold transition-all shrink-0 ${
              activeTab === tab.id
                ? "bg-raised text-accent shadow-card"
                : "text-ink-3 hover:text-ink"
            }`}
          >
            <tab.icon weight="bold" className="w-3.5 h-3.5" />
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* 4. Panels (lazy: only the active tab renders) */}
      <div>
        {activeTab === "resume" && (
          <div id="panel-resume" role="tabpanel" aria-labelledby="tab-resume" tabIndex={0} className="rounded-md2 bg-inset border border-line p-4">
            <div className="flex items-center justify-between gap-2 pb-2.5 mb-3 border-b border-line">
              <span className="text-2xs text-ink-3 font-mono">Markdown preview of the rewritten resume</span>
              {renderCopyRow("resume", result.optimized_resume)}
            </div>

            {canDiff && (
              <div className="mb-3">
                <button
                  type="button"
                  onClick={() => setShowDiff((s) => !s)}
                  aria-expanded={showDiff}
                  aria-controls="resume-diff"
                  className="flex items-center gap-1.5 text-2xs font-medium text-accent hover:text-accent-strong transition-colors px-2 py-1 rounded-sm2 bg-accent-tint border border-accent/20"
                >
                  <ArrowsLeftRight weight="bold" className="w-3.5 h-3.5" />
                  {showDiff ? "Hide before/after" : "Compare with original"}
                </button>
              </div>
            )}
            {showDiff && canDiff && (
              <div id="resume-diff" className="animate-in rounded-md2 bg-surface border border-line p-3 mb-4">
                <DiffView original={originalResume!} optimized={result.optimized_resume} />
              </div>
            )}

            <Markdown>{result.optimized_resume}</Markdown>

            {result.feedback && (
              <div className="mt-4 pt-3 border-t border-line">
                <div className="flex items-center gap-1.5 mb-1">
                  <Lightbulb weight="bold" className="w-3.5 h-3.5 text-accent" />
                  <span className="text-2xs font-semibold uppercase tracking-wide text-ink-2">Reviewer notes</span>
                </div>
                <p className="text-2xs text-ink-2 leading-relaxed whitespace-pre-line">{result.feedback}</p>
              </div>
            )}
          </div>
        )}

        {activeTab === "cover" && (
          <div id="panel-cover" role="tabpanel" aria-labelledby="tab-cover" tabIndex={0} className="rounded-md2 bg-inset border border-line p-4">
            <div className="flex items-center justify-between gap-2 pb-2.5 mb-3 border-b border-line">
              <span className="text-2xs text-ink-3 font-mono">Cover letter generated for this role</span>
              {renderCopyRow("cover", result.cover_letter)}
            </div>
            <Markdown>{result.cover_letter}</Markdown>
          </div>
        )}

        {activeTab === "ats" && (
          <div id="panel-ats" role="tabpanel" aria-labelledby="tab-ats" tabIndex={0} className="rounded-md2 bg-inset border border-line p-4">
            <div className="flex items-center justify-between gap-2 pb-2.5 mb-3 border-b border-line">
              <span className="text-2xs text-ink-3 font-mono">Keyword coverage report (deterministic)</span>
              {renderCopyRow("ats", result.ats_result)}
            </div>
            <Markdown className="font-mono">{result.ats_result}</Markdown>
          </div>
        )}

        {activeTab === "interview" && (
          <div id="panel-interview" role="tabpanel" aria-labelledby="tab-interview" tabIndex={0} className="rounded-md2 bg-inset border border-line p-4">
            <div className="flex items-center justify-between gap-2 pb-2.5 mb-3 border-b border-line">
              <span className="text-2xs text-ink-3 font-mono">Role-specific questions with STAR prompts</span>
              {renderCopyRow("interview", result.interview_questions)}
            </div>
            <Markdown>{result.interview_questions}</Markdown>
          </div>
        )}
      </div>
    </div>
  );
};
