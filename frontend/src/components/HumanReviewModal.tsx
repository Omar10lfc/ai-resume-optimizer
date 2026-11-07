import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Sparkle,
  ShieldCheck,
  Lightbulb,
  X,
  ArrowCounterClockwise,
  Notepad,
  Trash,
} from "@phosphor-icons/react";
import { parseGapSkills, GROUP_ORDER, GROUP_LABELS, type GapGroup } from "../lib/gaps";

/**
 * Step 1 → step 2 checkpoint UI.
 *
 * Behavior contract with App.tsx is unchanged: props stay
 * (missingSkills, userNotes, setUserNotes, onSubmit, isProcessing).
 * setUserNotes is invoked on submit with the composed review string built
 * from free-text notes + kept chips (with per-chip context) + an explicit
 * "do not feature" line for deleted chips (composition documented in
 * lib/gaps.ts) — so deletes are honored without API/format changes.
 *
 * Keyboard (chips are buttons with global focus rings):
 *   Delete / Backspace — remove focused chip (one-level undo kept)
 *   U                  — restore last removed chip
 *   Enter              — rename focused chip inline
 *   N                  — open the note field for the focused chip
 *   Esc                — close the inline editor or note field
 */

interface HumanReviewModalProps {
  missingSkills: string;
  userNotes: string;
  setUserNotes: (val: string) => void;
  /**
   * Receives the COMPOSED notes string. Passing it as an argument (instead of
   * letting App read `userNotes` state) is deliberate: setUserNotes here is
   * async, so reading state in the same tick submits the stale pre-edit value
   * — live-run evidence, Nov 5 2026 trace 520188c7: the composed gap list was
   * silently dropped and step 2 ran with empty notes.
   */
  onSubmit: (notes: string) => void;
  isProcessing: boolean;
}

interface EditorChip {
  id: string;
  label: string;
  group: GapGroup;
  note?: string;
  removed: boolean;
}

export const HumanReviewModal: React.FC<HumanReviewModalProps> = ({
  missingSkills,
  userNotes,
  setUserNotes,
  onSubmit,
  isProcessing,
}) => {
  const [chips, setChips] = useState<EditorChip[]>(() =>
    parseGapSkills(missingSkills).map((g) => ({ ...g, removed: false }))
  );
  // Re-parse only when THE LIST ACTUALLY CHANGES (fresh Step 1 run), using the
  // render-phase derived-state idiom React documents for prop-derived lists.
  const [freeNotes, setFreeNotes] = useState(userNotes);
  const [lastRemoved, setLastRemoved] = useState<EditorChip | null>(null);
  const [lastRaw, setLastRaw] = useState(missingSkills);
  if (missingSkills !== lastRaw) {
    setLastRaw(missingSkills);
    setChips(parseGapSkills(missingSkills).map((g) => ({ ...g, removed: false })));
    setFreeNotes(userNotes); // fresh run: adopt whatever App holds (reset ⇒ "")
    setLastRemoved(null);
  }

  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingValue, setEditingValue] = useState("");
  const [noteId, setNoteId] = useState<string | null>(null);
  const [noteValue, setNoteValue] = useState("");
  const editInputRef = useRef<HTMLInputElement>(null);
  const noteInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editingId) editInputRef.current?.focus();
  }, [editingId]);
  useEffect(() => {
    if (noteId) noteInputRef.current?.focus();
  }, [noteId]);

  const grouped = useMemo(() => {
    const map = new Map<GapGroup, EditorChip[]>();
    for (const group of GROUP_ORDER) map.set(group, []);
    for (const chip of chips) {
      if (!chip.removed) map.get(chip.group)?.push(chip);
    }
    for (const group of GROUP_ORDER) {
      map.get(group)!.sort((a, b) => a.label.localeCompare(b.label));
    }
    return map;
  }, [chips]);

  const activeCount = chips.filter((c) => !c.removed).length;
  const removedCount = chips.length - activeCount;
  const noteCount = chips.filter((c) => !c.removed && (c.note || "").trim()).length;

  const handleRemove = useCallback(
    (id: string) => {
      const target = chips.find((c) => c.id === id);
      if (!target || target.removed) return;
      setLastRemoved(target);
      setChips((prev) => prev.map((c) => (c.id === id ? { ...c, removed: true } : c)));
    },
    [chips]
  );

  const handleUndoLast = useCallback(() => {
    if (!lastRemoved) return;
    setChips((prev) => prev.map((c) => (c.id === lastRemoved.id ? { ...c, removed: false } : c)));
    setLastRemoved(null);
  }, [lastRemoved]);

  const handleRestore = useCallback((id: string) => {
    setLastRemoved(null);
    setChips((prev) => prev.map((c) => (c.id === id ? { ...c, removed: false } : c)));
  }, []);

  const commitEdit = useCallback(() => {
    if (!editingId) return;
    const value = editingValue.trim().slice(0, 120);
    setChips((prev) => prev.map((c) => (c.id === editingId ? { ...c, label: value || c.label } : c)));
    setEditingId(null);
    setEditingValue("");
  }, [editingId, editingValue]);

  const commitNote = useCallback(() => {
    if (!noteId) return;
    const value = noteValue.trim().slice(0, 300);
    setChips((prev) => prev.map((c) => (c.id === noteId ? { ...c, note: value || undefined } : c)));
    setNoteId(null);
    setNoteValue("");
  }, [noteId, noteValue]);

  const handleSubmit = () => {
    const kept = chips.filter((c) => !c.removed);
    const removedLabels = chips.filter((c) => c.removed).map((c) => c.label);
    const parts: string[] = [];
    if (freeNotes.trim()) parts.push(freeNotes.trim());
    for (const chip of kept) {
      const note = (chip.note || "").trim();
      parts.push(note ? `- ${chip.label}: ${note}` : `- ${chip.label}`);
    }
    if (removedLabels.length > 0) {
      parts.push(
        `Do NOT feature the following: ${removedLabels.join(", ")}. Do not add them unless the resume already proves them.`
      );
    }
    const composed = parts.join("\n");
    // Sync state (for retries/reload persistence) AND pass the string through
    // directly — the submit must never depend on the not-yet-committed state.
    setUserNotes(composed);
    onSubmit(composed);
  };

  const chipKeyboard = (chip: EditorChip) => (e: React.KeyboardEvent) => {
    if (editingId === chip.id || noteId === chip.id) return;
    if (e.key === "Delete" || e.key === "Backspace") {
      e.preventDefault();
      handleRemove(chip.id);
    } else if (e.key.toLowerCase() === "u") {
      e.preventDefault();
      handleUndoLast();
    } else if (e.key === "Enter") {
      e.preventDefault();
      setEditingId(chip.id);
      setEditingValue(chip.label);
    } else if (e.key.toLowerCase() === "n") {
      e.preventDefault();
      setNoteId(chip.id);
      setNoteValue(chip.note || "");
    }
  };

  return (
    <section className="panel p-4 mb-6" aria-label="Review gap analysis">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 pb-3 mb-4 border-b border-line">
        <div>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-ok" aria-hidden="true" />
            <span className="text-2xs font-semibold uppercase tracking-wide text-accent">
              Step 1 complete
            </span>
          </div>
          <h3 className="text-sm font-semibold text-ink mt-1">Review the gaps, add honest context</h3>
          <p className="text-2xs text-ink-3 mt-0.5">
            {activeCount} gap{activeCount === 1 ? "" : "s"} detected · {noteCount} with context ·{" "}
            {removedCount} removed
          </p>
        </div>
        <div className="flex items-center gap-1.5 text-2xs font-medium text-ok bg-ok-tint border border-ok/25 px-2.5 py-1.5 rounded-md2 self-start">
          <ShieldCheck weight="bold" className="w-3.5 h-3.5 shrink-0" />
          <span>Truthfulness guardrail active</span>
        </div>
      </div>

      {/* Grouped chips */}
      <div className="space-y-4">
        {GROUP_ORDER.map((group) => {
          const list = grouped.get(group)!;
          if (list.length === 0) return null;
          return (
            <div key={group}>
              <h4 className="flex items-center gap-1.5 text-2xs font-semibold text-ink-2 mb-2">
                <Lightbulb
                  weight="bold"
                  className={`w-3 h-3 ${
                    group === "tool" ? "text-accent" : group === "concept" ? "text-warn" : "text-ok"
                  }`}
                />
                <span>{GROUP_LABELS[group]}</span>
                <span className="text-ink-3 font-mono">({list.length})</span>
              </h4>
              <div role="list" className="flex flex-wrap gap-2">
                {list.map((chip) => {
                  const isEditing = editingId === chip.id;
                  const isNoting = noteId === chip.id;
                  return (
                    <div key={chip.id} className="min-w-0" role="listitem">
                      <div className="flex items-center gap-1 px-2 py-1 rounded-full bg-raised border border-line max-w-full transition-colors hover:border-line-strong">
                        {isEditing ? (
                          <input
                            ref={editInputRef}
                            value={editingValue}
                            onChange={(e) => setEditingValue(e.target.value)}
                            onBlur={commitEdit}
                            onKeyDown={(e) => {
                              if (e.key === "Enter") {
                                e.preventDefault();
                                commitEdit();
                              }
                              if (e.key === "Escape") {
                                e.preventDefault();
                                setEditingId(null);
                                setEditingValue("");
                              }
                            }}
                            className="flex-1 min-w-0 bg-transparent text-xs text-ink outline-none px-1"
                            aria-label={`Rename ${chip.label}`}
                          />
                        ) : (
                          <button
                            type="button"
                            onKeyDown={chipKeyboard(chip)}
                            onClick={() => {
                              setEditingId(chip.id);
                              setEditingValue(chip.label);
                            }}
                            className="flex-1 min-w-0 text-left text-xs text-ink px-1 py-0.5 leading-snug line-clamp-2 [overflow-wrap:anywhere]"
                            title={`${chip.label} — Enter renames · N opens note · Delete removes`}
                          >
                            {chip.label}
                          </button>
                        )}
                        <button
                          type="button"
                          onClick={() => {
                            if (isNoting) {
                              setNoteId(null);
                              setNoteValue("");
                            } else {
                              setNoteId(chip.id);
                              setNoteValue(chip.note || "");
                            }
                          }}
                          aria-label={`${isNoting ? "Close note for" : "Add note to"} ${chip.label}`}
                          className={`p-0.5 rounded-sm2 shrink-0 transition-colors ${
                            chip.note ? "text-accent" : "text-ink-3 hover:text-accent"
                          }`}
                        >
                          <Notepad weight={chip.note ? "fill" : "regular"} className="w-3.5 h-3.5" />
                        </button>
                        <button
                          type="button"
                          onClick={() => handleRemove(chip.id)}
                          aria-label={`Remove ${chip.label} (Delete key)`}
                          className="p-0.5 rounded-sm2 text-ink-3 hover:text-danger transition-colors shrink-0"
                        >
                          <X weight="bold" className="w-3.5 h-3.5" />
                        </button>
                      </div>

                      {isNoting && (
                        <input
                          ref={noteInputRef}
                          value={noteValue}
                          onChange={(e) => setNoteValue(e.target.value)}
                          onBlur={commitNote}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") {
                              e.preventDefault();
                              commitNote();
                            }
                            if (e.key === "Escape") {
                              e.preventDefault();
                              setNoteId(null);
                              setNoteValue("");
                            }
                          }}
                          placeholder="Where or how have you used this?"
                          aria-label={`Context note for ${chip.label}`}
                          className="w-full mt-1 px-2.5 py-1.5 rounded-md2 bg-inset border border-line text-2xs text-ink placeholder-ink-3 focus:border-accent outline-none"
                        />
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}

        {activeCount === 0 && (
          <div
            className="p-4 rounded-md2 bg-inset border border-line text-xs text-ink-2 leading-relaxed"
            role="status"
          >
            No skill gaps flagged. The optimizer will work from the job description and your resume as-is.
          </div>
        )}
      </div>

      {/* Removed tray with undo affordances */}
      {removedCount > 0 && (
        <div className="mt-4 pt-3 border-t border-line">
          <div className="flex items-center justify-between gap-2 mb-2">
            <span className="text-2xs text-ink-3 font-medium flex items-center gap-1.5">
              <Trash weight="bold" className="w-3 h-3" />
              Removed ({removedCount}) — will not be featured
            </span>
            {lastRemoved && (
              <button
                type="button"
                onClick={handleUndoLast}
                className="flex items-center gap-1 text-2xs text-accent hover:text-accent-strong font-medium px-1.5 py-0.5 rounded-sm2 transition-colors"
              >
                <ArrowCounterClockwise weight="bold" className="w-3 h-3" />
                Undo last (U)
              </button>
            )}
          </div>
          <div role="list" className="flex flex-wrap gap-1.5">
            {chips
              .filter((c) => c.removed)
              .map((chip) => (
                <button
                  key={chip.id}
                  type="button"
                  onClick={() => handleRestore(chip.id)}
                  aria-label={`Restore ${chip.label}`}
                  className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-inset border border-line text-2xs text-ink-3 line-through hover:text-ink hover:border-line-strong transition-colors"
                >
                  {chip.label}
                  <ArrowCounterClockwise weight="bold" className="w-3 h-3" />
                </button>
              ))}
          </div>
        </div>
      )}

      {/* Free-form notes */}
      <div className="mt-5">
        <label htmlFor="human-notes-textarea" className="field-label">
          Anything else the optimizer should know?
        </label>
        <textarea
          id="human-notes-textarea"
          rows={4}
          value={freeNotes}
          onChange={(e) => setFreeNotes(e.target.value)}
          placeholder="Projects, courses, or hands-on context for the gaps above..."
          disabled={isProcessing}
          className="input font-mono resize-none"
        />
      </div>

      <p className="text-2xs text-ink-3 mt-3 font-mono leading-relaxed">
        Keys: Delete removes · U restores · Enter renames · N opens a note
      </p>

      <div className="pt-4 flex justify-end">
        <button
          type="button"
          onClick={handleSubmit}
          disabled={isProcessing}
          className="btn btn-primary w-full sm:w-auto"
        >
          {isProcessing ? (
            <>
              <span className="w-3.5 h-3.5 border-2 border-accent-contrast border-t-transparent rounded-full animate-spin-slow" />
              <span>Generating documents...</span>
            </>
          ) : (
            <>
              <Sparkle weight="bold" className="w-4 h-4" />
              <span>Step 2 — Generate all documents</span>
            </>
          )}
        </button>
      </div>
    </section>
  );
};
