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
  ShieldWarning,
  Prohibit,
  Target,
  X,
  Lightning,
} from "@phosphor-icons/react";
import type { OptimizationResult, RequirementMatch } from "../types";
import { resolveArtifactUrl } from "../api";
import { ScoreGauge } from "./ScoreGauge";
import { DiffView } from "./DiffView";
import { Markdown } from "./Markdown";

type TabId = "resume" | "cover" | "ats" | "fit" | "interview" | "integrity" | "usage";

interface ResultsDashboardProps {
  result: OptimizationResult;
  /** Powers the before/after diff view: paste mode passes the client-held
   *  text, uploads rely on the server echo (PB3 include_diff=original). */
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

  // The Integrity tab only exists when this run actually has integrity
  // content (QW2/QW3): removals, declined requests, or metric findings
  // (resume path + the cover letter's own scan, live run 22e943c6).
  const integrityList = [
    ...(result.removals ?? []),
    ...(result.declined_requests ?? []),
    ...(result.unevidenced_metrics ?? []),
    ...(result.cover_letter_metrics ?? []),
  ];
  // Tabs beyond the core four exist only when this run has their content:
  // Fit (MR2 requirement table), Integrity (QW2/QW3 channels), Usage (MR5
  // per-task token accounting).
  const fitRows = (result.requirement_matches ?? []) as RequirementMatch[];
  const taskUsage = result.task_usage ?? {};
  const usageRows = Object.entries(taskUsage);
  const usageTotals = usageRows.length
    ? usageRows.reduce(
        (acc, [, s]) => ({
          prompt_tokens: acc.prompt_tokens + s.prompt_tokens,
          completion_tokens: acc.completion_tokens + s.completion_tokens,
          total_tokens: acc.total_tokens + s.total_tokens,
          llm_calls: acc.llm_calls + s.llm_calls,
        }),
        { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0, llm_calls: 0 })
    : null;
  const effectiveTabs: Array<{ id: TabId; label: string; icon: React.ElementType }> = [
    ...TABS,
  ];
  if (fitRows.length > 0) {
    effectiveTabs.splice(3, 0, { id: "fit" as TabId, label: "Fit", icon: Target });
  }
  if (integrityList.length > 0) {
    effectiveTabs.push({ id: "integrity" as TabId, label: "Integrity", icon: ShieldWarning });
  }
  if (usageRows.length > 0) {
    effectiveTabs.push({ id: "usage" as TabId, label: "Usage", icon: Lightning });
  }

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
    const idx = effectiveTabs.findIndex((t) => t.id === activeTab);
    const move = (dir: number) => {
      e.preventDefault();
      const next = effectiveTabs[(idx + dir + effectiveTabs.length) % effectiveTabs.length];
      setActiveTab(next.id);
      tabRefs.current.get(next.id)?.focus();
    };
    if (e.key === "ArrowRight") move(1);
    else if (e.key === "ArrowLeft") move(-1);
    else if (e.key === "Home") {
      e.preventDefault();
      setActiveTab(effectiveTabs[0].id);
      tabRefs.current.get(effectiveTabs[0].id)?.focus();
    } else if (e.key === "End") {
      e.preventDefault();
      const last = effectiveTabs[effectiveTabs.length - 1];
      setActiveTab(last.id);
      tabRefs.current.get(last.id)?.focus();
    }
  };

  const canDiff = Boolean(originalResume && originalResume.trim().length > 40);

  const fitCounts = fitRows.length
    ? [
        fitRows.filter((r) => r.status === "covered").length,
        fitRows.filter((r) => r.status === "partial").length,
        fitRows.filter((r) => r.status === "gap").length,
      ]
    : null;
  const fitCopy = fitRows.length
    ? `Requirement fit (${fitCounts![0]}/${fitRows.length} fully covered):\n` +
      fitRows.map((r) => `[${r.status.toUpperCase()}] ${r.requirement} — ${r.reason}`).join("\n")
    : "";
  const usageCopy = usageRows.length
    ? `Per-task token usage (${usageTotals!.llm_calls} LLM calls, ${usageTotals!.total_tokens} tokens):\n` +
      usageRows.map(([task, s]) =>
        `${task}: ${s.llm_calls} ${s.llm_calls === 1 ? "call" : "calls"}, ${s.prompt_tokens} in / ${s.completion_tokens} out`).join("\n")
    : "";

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
        {effectiveTabs.map((tab) => (
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

        {activeTab === "fit" && (
          <div id="panel-fit" role="tabpanel" aria-labelledby="tab-fit" tabIndex={0} className="rounded-md2 bg-inset border border-line p-4">
            <div className="flex items-center justify-between gap-2 pb-2.5 mb-3 border-b border-line">
              <span className="text-2xs text-ink-3 font-mono">Requirement fit — classified in code, never by a model</span>
              {renderCopyRow("fit", fitCopy)}
            </div>

            <p className="text-2xs text-ink-3 leading-relaxed mb-3">
              Each requirement the posting states, judged in code against the shipped resume.
              Read it next to the ATS percentage: a high keyword score missing whole requirements shows up here as gap rows.
            </p>
            {fitCounts && (
              <div className="flex items-center gap-2 mb-3">
                <span className="px-2 py-0.5 rounded-sm2 bg-ok-tint border border-ok/25 text-2xs text-ok font-semibold">{fitCounts[0]} covered</span>
                <span className="px-2 py-0.5 rounded-sm2 bg-warn-tint border border-warn/25 text-2xs text-warn font-semibold">{fitCounts[1]} partial</span>
                <span className="px-2 py-0.5 rounded-sm2 bg-danger-tint border border-danger/25 text-2xs text-danger font-semibold">{fitCounts[2]} gap</span>
              </div>
            )}
            <ul className="space-y-2">
              {fitRows.map((row, i) => (
                <li key={`fit-${i}`} className="rounded-md2 bg-surface border border-line p-2.5">
                  <div className="flex items-start gap-2">
                    <span
                      aria-label={row.status}
                      className={`flex items-center gap-1 px-1.5 py-0.5 rounded-sm2 border text-2xs font-semibold uppercase tracking-wide shrink-0 ${
                        row.status === "covered"
                          ? "bg-ok-tint border-ok/25 text-ok"
                          : row.status === "partial"
                            ? "bg-warn-tint border-warn/25 text-warn"
                            : "bg-danger-tint border-danger/25 text-danger"
                      }`}
                    >
                      {row.status === "covered"
                        ? <Check weight="bold" className="w-3 h-3" />
                        : row.status === "partial"
                          ? <Prohibit weight="bold" className="w-3 h-3" />
                          : <X weight="bold" className="w-3 h-3" />}
                      {row.status}
                    </span>
                    <span className="flex-1 min-w-0 text-2xs text-ink leading-relaxed">{row.requirement}</span>
                  </div>
                  {row.reason && (
                    <p className="mt-1.5 text-2xs text-ink-2 leading-relaxed">{row.reason}</p>
                  )}
                </li>
              ))}
            </ul>

            {Object.keys(result.prompt_versions ?? {}).length > 0 && (
              <div className="pt-3 mt-3 border-t border-line">
                <span className="text-2xs text-ink-3 font-mono leading-relaxed break-all">
                  prompts: {Object.values(result.prompt_versions ?? {}).join(" · ")}
                </span>
              </div>
            )}
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

        {activeTab === "integrity" && (
          <div id="panel-integrity" role="tabpanel" aria-labelledby="tab-integrity" tabIndex={0} className="rounded-md2 bg-inset border border-line p-4">
            <div className="mb-4">
              <div className="flex items-center gap-1.5 mb-1">
                <ShieldWarning weight="bold" className="w-3.5 h-3.5 text-warn" />
                <span className="text-2xs font-semibold uppercase tracking-wide text-ink-2">Integrity report</span>
              </div>
              <p className="text-2xs text-ink-3 leading-relaxed">
                What was checked against your evidence (original resume + your notes) before this result shipped.
              </p>
            </div>

            {(result.removals?.length ?? 0) > 0 && (
              <div className="mb-4">
                <h4 className="text-2xs font-semibold text-ink mb-1.5">Removed deterministically</h4>
                <ul className="space-y-1">
                  {result.removals!.map((item, i) => (
                    <li key={`rm-${i}`} className="flex items-start gap-2 text-2xs text-warn leading-relaxed">
                      <span className="mt-[5px] w-1 h-1 rounded-full bg-warn shrink-0" aria-hidden="true" />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {(result.declined_requests?.length ?? 0) > 0 && (
              <div className="mb-4">
                <h4 className="text-2xs font-semibold text-ink mb-1.5 flex items-center gap-1.5">
                  <Prohibit weight="bold" className="w-3 h-3 text-ink-3" />
                  Requests the writer declined
                </h4>
                <p className="text-2xs text-ink-3 mb-1.5">Your notes or the gap list asked for these; the evidence does not show them, so they were left out instead of invented:</p>
                <ul className="space-y-1">
                  {result.declined_requests!.map((item, i) => (
                    <li key={`dc-${i}`} className="flex items-start gap-2 text-2xs text-ink-2 leading-relaxed">
                      <span className="mt-[5px] w-1 h-1 rounded-full bg-ink-3 shrink-0" aria-hidden="true" />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {(result.unevidenced_metrics?.length ?? 0) > 0 && (
              <div className="mb-4">
                <h4 className="text-2xs font-semibold text-ink mb-1.5">Numbers to verify</h4>
                <p className="text-2xs text-ink-3 mb-1.5">Still in the resume; the scrubber could not tie them to your evidence:</p>
                <div className="flex flex-wrap gap-1.5">
                  {result.unevidenced_metrics!.map((item, i) => (
                    <span key={`um-${i}`} className="px-2 py-0.5 rounded-sm2 bg-warn-tint border border-warn/25 text-2xs text-warn font-mono">{item}</span>
                  ))}
                </div>
              </div>
            )}

            {(result.cover_letter_metrics?.length ?? 0) > 0 && (
              <div className="mb-4">
                <h4 className="text-2xs font-semibold text-ink mb-1.5">Cover-letter numbers to verify</h4>
                <p className="text-2xs text-ink-3 mb-1.5">Still in the cover letter; neither your resume nor your notes contain them (live run 22e943c6 fabricated "70%"/"45%" this way). Verify before sending:</p>
                <div className="flex flex-wrap gap-1.5">
                  {result.cover_letter_metrics!.map((item, i) => (
                    <span key={`clm-${i}`} className="px-2 py-0.5 rounded-sm2 bg-warn-tint border border-warn/25 text-2xs text-warn font-mono">{item}</span>
                  ))}
                </div>
              </div>
            )}

            {Object.keys(result.prompt_versions ?? {}).length > 0 && (
              <div className="pt-3 border-t border-line">
                <span className="text-2xs text-ink-3 font-mono leading-relaxed break-all">
                  prompts: {Object.values(result.prompt_versions ?? {}).join(" · ")}
                </span>
              </div>
            )}
          </div>
        )}

        {activeTab === "usage" && (
          <div id="panel-usage" role="tabpanel" aria-labelledby="tab-usage" tabIndex={0} className="rounded-md2 bg-inset border border-line p-4">
            <div className="flex items-center justify-between gap-2 pb-2.5 mb-3 border-b border-line">
              <span className="text-2xs text-ink-3 font-mono">Token accounting per pipeline stage (from the provider's usage, in code)</span>
              {renderCopyRow("usage", usageCopy)}
            </div>

            <p className="text-2xs text-ink-3 leading-relaxed mb-3">
              What each stage actually spent on this run. Task names match the prompt stamps shown on the Fit tab, so a suspicious cost points straight at the stage (and prompt version) that ran it.
            </p>
            <ul className="space-y-1.5">
              {usageRows.map(([task, s]) => (
                <li key={`usage-${task}`} className="flex items-center gap-2 px-3 py-2 rounded-md2 bg-surface border border-line text-2xs">
                  <span className="flex-1 min-w-0 font-mono text-ink truncate">{task}</span>
                  <span className="font-mono text-ink-3 shrink-0">{s.llm_calls} {s.llm_calls === 1 ? "call" : "calls"}</span>
                  <span className="font-mono text-ink-3 shrink-0">{s.prompt_tokens} in / {s.completion_tokens} out</span>
                  <span className="font-mono text-ink font-semibold shrink-0 w-16 text-right">{s.total_tokens}</span>
                </li>
              ))}
            </ul>
            {usageTotals && (
              <div className="pt-3 mt-3 border-t border-line flex items-center gap-3 text-2xs">
                <span className="text-ink-2 font-semibold uppercase tracking-wide flex-1">Total</span>
                <span className="font-mono text-ink-3 shrink-0">{usageTotals.llm_calls} {usageTotals.llm_calls === 1 ? "call" : "calls"}</span>
                <span className="font-mono text-ink-3 shrink-0">{usageTotals.prompt_tokens} in / {usageTotals.completion_tokens} out</span>
                <span className="font-mono text-ink font-semibold shrink-0 w-16 text-right">{usageTotals.total_tokens}</span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
