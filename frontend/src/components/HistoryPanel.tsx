import React, { useCallback, useEffect, useState } from "react";
import {
  ArrowClockwise,
  CaretDown,
  Trash,
} from "@phosphor-icons/react";
import {
  deleteApplication,
  fetchApplication,
  fetchApplications,
} from "../api";
import type { ApplicationDetail, ApplicationSummary } from "../types";

function fmtTime(unixSeconds: number | null): string {
  if (!unixSeconds) return "—";
  return new Date(unixSeconds * 1000).toLocaleString();
}

function scoreClass(score: number): string {
  if (score >= 80) return "bg-ok-tint border-ok/25 text-ok";
  if (score >= 60) return "bg-warn-tint border-warn/25 text-warn";
  return "bg-danger-tint border-danger/25 text-danger";
}

/** Fit counts for one recorded run's requirement table. */
function fitCounts(
  rows: Array<{ status: string }>
): [number, number, number] | null {
  if (!rows.length) return null;
  const count = (s: string) => rows.filter((r) => r.status === s).length;
  return [count("covered"), count("partial"), count("gap")];
}

const ApplicationRuns: React.FC<{ detail: ApplicationDetail }> = ({ detail }) => (
  <ul className="space-y-2">
    {detail.runs.map((run) => {
      const counts = fitCounts(run.result?.requirement_matches ?? []);
      return (
        <li
          key={run.id}
          className="rounded-md2 bg-surface border border-line p-2.5"
        >
          <div className="flex items-center gap-2 flex-wrap">
            <span
              className={`px-2 py-0.5 rounded-sm2 border text-2xs font-semibold ${scoreClass(run.score)}`}
            >
              {Math.round(run.score)}/100
            </span>
            <span className="px-2 py-0.5 rounded-sm2 bg-inset border border-line text-2xs text-ink-2 font-mono">
              ATS {Math.round(run.ats_percentage)}%
            </span>
            <span className="px-2 py-0.5 rounded-sm2 bg-inset border border-line text-2xs text-ink-2 font-mono">
              LLM {Math.round(run.llm_score)}
            </span>
            {run.review_failed === 1 && (
              <span className="px-2 py-0.5 rounded-sm2 bg-warn-tint border border-warn/25 text-2xs text-warn font-semibold">
                review salvaged
              </span>
            )}
            <span className="text-2xs text-ink-3 font-mono ml-auto">
              {fmtTime(run.created_at)}
            </span>
          </div>
          {counts && (
            <div className="flex items-center gap-2 mt-2">
              <span className="px-2 py-0.5 rounded-sm2 bg-ok-tint border border-ok/25 text-2xs text-ok font-semibold">
                {counts[0]} covered
              </span>
              <span className="px-2 py-0.5 rounded-sm2 bg-warn-tint border border-warn/25 text-2xs text-warn font-semibold">
                {counts[1]} partial
              </span>
              <span className="px-2 py-0.5 rounded-sm2 bg-danger-tint border border-danger/25 text-2xs text-danger font-semibold">
                {counts[2]} gap
              </span>
            </div>
          )}
        </li>
      );
    })}
  </ul>
);

/**
 * BB1 Phase A: the application history (persistent profile store).
 * Runs are auto-recorded server-side on every completed run; this panel
 * reviews and deletes them. Collapsed by default; fetches on first expand.
 */
export const HistoryPanel: React.FC = () => {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apps, setApps] = useState<ApplicationSummary[] | null>(null);
  const [expanded, setExpanded] = useState<number | null>(null);
  const [detail, setDetail] = useState<ApplicationDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setApps(await fetchApplications());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load history.");
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch on first expand (not on mount - the landing page stays quiet).
  useEffect(() => {
    if (open && apps === null && !loading) {
      refresh();
    }
  }, [open, apps, loading, refresh]);

  const toggleRun = useCallback(async (id: number) => {
    if (expanded === id) {
      setExpanded(null);
      setDetail(null);
      return;
    }
    setExpanded(id);
    setDetailLoading(true);
    setDetail(null);
    try {
      setDetail(await fetchApplication(id));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load application.");
    } finally {
      setDetailLoading(false);
    }
  }, [expanded]);

  const remove = useCallback(async (id: number) => {
    try {
      await deleteApplication(id);
      setDetail(null);
      setExpanded(null);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to delete.");
    }
  }, [refresh]);

  return (
    <section
      aria-label="Application history"
      className="mt-6 rounded-md2 bg-inset border border-line overflow-hidden"
    >
      <div className="w-full flex items-center gap-2 px-4 py-2.5 border-b border-line">
        <button
          type="button"
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
          className="flex items-center gap-2 flex-1 min-w-0 text-left hover:bg-surface/60 transition-colors rounded-sm2"
        >
          <CaretDown
            weight="bold"
            className={`w-3.5 h-3.5 text-ink-3 transition-transform ${open ? "" : "-rotate-90"}`}
          />
          <span className="text-2xs font-semibold uppercase tracking-wide text-ink-2">
            Application history
          </span>
          <span className="text-2xs text-ink-3 font-mono truncate">
            auto-recorded from completed runs
          </span>
        </button>
        {open && (
          <button
            type="button"
            aria-label="Refresh history"
            onClick={refresh}
            className="p-1 rounded-sm2 text-ink-3 hover:text-ink hover:bg-surface shrink-0"
          >
            <ArrowClockwise className="w-3.5 h-3.5" />
          </button>
        )}
      </div>

      {open && (
        <div className="px-4 py-3">
          {loading && (
            <p className="text-2xs text-ink-3 font-mono">Loading…</p>
          )}
          {error && (
            <p className="text-2xs text-danger font-mono" role="alert">
              {error}
            </p>
          )}
          {!loading && !error && apps !== null && apps.length === 0 && (
            <p className="text-2xs text-ink-3 leading-relaxed">
              No applications recorded yet. Every completed optimization run
              lands here automatically (with its scores and requirement-fit
              history). Disable recording with{" "}
              <code className="font-mono">PROFILE_DB_PATH=off</code>.
            </p>
          )}
          {!loading && apps !== null && apps.length > 0 && (
            <ul className="space-y-2">
              {apps.map((app) => (
                <li
                  key={app.id}
                  className="rounded-md2 bg-surface border border-line"
                >
                  <div className="flex items-center gap-2 flex-wrap p-2.5">
                    <button
                      type="button"
                      aria-expanded={expanded === app.id}
                      onClick={() => toggleRun(app.id)}
                      className="flex-1 min-w-0 text-left"
                    >
                      <span className="block text-2xs text-ink font-medium truncate">
                        {app.title}
                      </span>
                      <span className="block text-2xs text-ink-3 font-mono mt-0.5">
                        {app.attempts} attempt{app.attempts === 1 ? "" : "s"}
                        {" · "}best{" "}
                        {app.best_score != null
                          ? `${Math.round(app.best_score)}/100`
                          : "—"}
                        {" · "}{fmtTime(app.updated_at)}
                      </span>
                    </button>
                    <button
                      type="button"
                      aria-label={`Delete application ${app.title}`}
                      onClick={() => remove(app.id)}
                      className="p-1.5 rounded-sm2 text-ink-3 hover:text-danger hover:bg-danger-tint shrink-0"
                    >
                      <Trash className="w-3.5 h-3.5" />
                    </button>
                  </div>
                  {expanded === app.id && (
                    <div className="px-2.5 pb-2.5">
                      {detailLoading && (
                        <p className="text-2xs text-ink-3 font-mono py-1">
                          Loading runs…
                        </p>
                      )}
                      {detail && detail.runs.length > 0 && (
                        <ApplicationRuns detail={detail} />
                      )}
                      {detail && detail.runs.length === 0 && (
                        <p className="text-2xs text-ink-3 font-mono py-1">
                          No runs recorded under this application.
                        </p>
                      )}
                    </div>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  );
};
