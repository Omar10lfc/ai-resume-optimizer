import React, { useMemo } from "react";
import { ArrowDown, ArrowsLeftRight } from "@phosphor-icons/react";

/**
 * Before/after view for the rewritten resume (paste mode only: upload mode
 * never holds the original text client-side, since PDF/DOCX parsing happens
 * server-side and the optimizer result carries no copy of the original).
 *
 * Strategy: line-level comparison.
 *   1. Split both documents into statement lines (resume bullets/lines).
 *   2. Longest-common-subsequence over normalized tokens pairs identical
 *      rewrites, so untouched content isn't shown as changed noise.
 *   3. Remaining lines pair up by similarity; close pairs render as
 *      "changed" (before → after), the rest as added/removed.
 * Similarity pairs are approximate by nature: they surface intent, they do
 * not claim byte-precision, and they degrade gracefully to "all changed"
 * when the rewrite is genuinely fresh.
 */

type DiffRow =
  | { type: "same"; text: string }
  | { type: "changed"; before: string; after: string; similarity: number }
  | { type: "added"; text: string }
  | { type: "removed"; text: string };

const MIN_SIMILARITY = 0.42;

function tokenize(line: string): string[] {
  return line
    .toLowerCase()
    .replace(/^[^\p{L}\p{N}]+/u, "")
    .split(/[^\p{L}\p{N}]+/u)
    .filter(Boolean);
}

/** Lenmax-normalized 2-gram overlap (Dice coefficient) on word tokens. */
function similarity(a: string, b: string): number {
  const ta = tokenize(a);
  const tb = tokenize(b);
  if (ta.length === 0 || tb.length === 0) return 0;
  const gramsA = new Set<string>();
  for (let i = 0; i < ta.length - 1; i++) gramsA.add(`${ta[i]}|${ta[i + 1]}`);
  if (ta.length === 1) gramsA.add(ta[0]);
  const gramsB = new Set<string>();
  for (let i = 0; i < tb.length - 1; i++) gramsB.add(`${tb[i]}|${tb[i + 1]}`);
  if (tb.length === 1) gramsB.add(tb[0]);
  let hits = 0;
  for (const g of gramsA) if (gramsB.has(g)) hits++;
  return (2 * hits) / (gramsA.size + gramsB.size);
}

const BULLET_RE = /^\s*(?:[-•*·]|\d+[.)])\s*/;

function splitLines(doc: string): string[] {
  return (doc || "")
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0 && !/^[#>]{1,3}$/.test(l));
}

function computeDiff(original: string, optimized: string): DiffRow[] {
  const A = splitLines(original);
  const B = splitLines(optimized);
  // Guardrail: pathological sizes crash the LCS table (n*m allocation).
  if (A.length * B.length > 40_000) {
    return [
      ...A.map((text) => ({ type: "same" as const, text })),
      ...B.map((text) => ({ type: "added" as const, text })),
    ];
  }

  // LCS over normalized bullet content so identical lines stay aligned even
  // when bullet markers change.
  const key = (l: string) => l.replace(BULLET_RE, "").toLowerCase();
  const n = A.length;
  const m = B.length;
  const dp: number[][] = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i][j] = key(A[i]) === key(B[j]) ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  // Walk the LCS; each step produces consumed ranges for A-side and B-side.
  const anchors: Array<{ a: number; b: number }> = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (key(A[i]) === key(B[j])) {
      anchors.push({ a: i++, b: j++ });
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      i++;
    } else {
      j++;
    }
  }
  // Fabricate terminal anchor so the tail slices don't fall off the edge.
  anchors.push({ a: n, b: m });

  const rows: DiffRow[] = [];
  let prevA = 0;
  let prevB = 0;
  for (const anchor of anchors) {
    const aRange = A.slice(prevA, anchor.a);
    const bRange = B.slice(prevB, anchor.b);
    // Greedy pairing on similarity within the unmatched block.
    const paired = new Set<number>();
    for (let ai = 0; ai < aRange.length; ai++) {
      let best = -1;
      let bestScore = 0;
      for (let bi = 0; bi < bRange.length; bi++) {
        if (paired.has(bi)) continue;
        const s = similarity(aRange[ai], bRange[bi]);
        if (s > bestScore) {
          bestScore = s;
          best = bi;
        }
      }
      if (best >= 0 && bestScore >= MIN_SIMILARITY) {
        paired.add(best);
        rows.push({ type: "changed", before: aRange[ai], after: bRange[best], similarity: bestScore });
      } else {
        rows.push({ type: "removed", text: aRange[ai] });
      }
    }
    for (let bi = 0; bi < bRange.length; bi++) {
      if (!paired.has(bi)) rows.push({ type: "added", text: bRange[bi] });
    }
    if (anchor.a < n && anchor.b < m) rows.push({ type: "same", text: A[anchor.a] });
    prevA = anchor.a + 1;
    prevB = anchor.b + 1;
  }
  return rows;
}

const RemovedRow: React.FC<{ text: string }> = ({ text }) => (
  <div className="flex items-start gap-2 px-3 py-1.5 bg-danger-tint rounded-sm2">
    <ArrowDown weight="bold" className="w-3.5 h-3.5 mt-0.5 text-danger shrink-0 rotate-180" />
    <span className="text-xs text-ink-2 leading-relaxed line-through decoration-danger/50">{text}</span>
  </div>
);

const AddedRow: React.FC<{ text: string }> = ({ text }) => (
  <div className="flex items-start gap-2 px-3 py-1.5 bg-ok-tint rounded-sm2">
    <ArrowDown weight="bold" className="w-3.5 h-3.5 mt-0.5 text-ok shrink-0" />
    <span className="text-xs text-ink leading-relaxed">{text}</span>
  </div>
);

export const DiffView: React.FC<{ original: string; optimized: string }> = ({ original, optimized }) => {
  const rows = useMemo(() => computeDiff(original, optimized), [original, optimized]);
  const stats = useMemo(() => {
    const changed = rows.filter((r) => r.type === "changed").length;
    const added = rows.filter((r) => r.type === "added").length;
    const removed = rows.filter((r) => r.type === "removed").length;
    return { changed, added, removed };
  }, [rows]);

  return (
    <div>
      <p className="text-2xs text-ink-3 mb-3 font-mono">
        {stats.changed} rewritten · {stats.added} added · {stats.removed} removed
      </p>
      <div className="space-y-1.5">
        {rows.map((row, idx) => {
          if (row.type === "changed") {
            return (
              <div key={idx} className="rounded-md2 border border-line overflow-hidden">
                <div className="flex items-start gap-2 px-3 py-1.5 bg-danger-tint">
                  <span className="w-3.5 shrink-0" aria-hidden="true" />
                  <span className="text-xs text-ink-2 leading-relaxed line-through decoration-danger/40">{row.before}</span>
                </div>
                <div className="flex items-start gap-2 px-3 py-1.5 bg-ok-tint border-t border-line/60">
                  <ArrowsLeftRight weight="bold" className="w-3.5 h-3.5 mt-0.5 text-ok shrink-0" />
                  <span className="text-xs text-ink leading-relaxed">{row.after}</span>
                </div>
              </div>
            );
          }
          if (row.type === "removed") return <RemovedRow key={idx} text={row.text} />;
          if (row.type === "added") return <AddedRow key={idx} text={row.text} />;
          return (
            <div key={idx} className="px-3 py-1">
              <span className="text-xs text-ink-3 leading-relaxed">{row.text}</span>
            </div>
          );
        })}
      </div>
      <p className="text-2xs text-ink-3 mt-3">
        Pairs are matched by meaning, not position — a large rewrite can read as "removed and added" rather than changed.
      </p>
    </div>
  );
};
