/**
 * Gap editor logic.
 *
 * The scanner sends missing skills as plain text (one bullet per line). The
 * step-2 endpoint consumes a single free-text `notes` field — it has no
 * structured field for accepted/rejected skills, and the backend's improver
 * prompt (agent/nodes/improver.py) incorporates notes as guidance context.
 *
 * Compose happens at submit time, so the UI can offer chips, notes, and
 * delete-undo freely while the payload phase stays one deterministic string:
 *
 *   1. Whatever free-text the user already typed (always preserved).
 *   2. Kept chips with context:          "- Docker: ran containers in coursework"
 *   3. Kept chips without context:       "- Terraform"
 *   4. Removed chips (single line):      "Do NOT feature: X, Y" — the improver
 *      is instructed by the server prompt to keep claims truthful, so this
 *      explicit instruction is how a "delete" is honored without API changes.
 */

export type GapGroup = "tool" | "concept" | "general";

export interface GapItem {
  id: string;
  label: string;
  group: GapGroup;
  note?: string;
}

const TOOL_KEYWORDS = [
  "docker", "kubernetes", "k8s", "terraform", "ansible", "jenkins", "gitlab",
  "github", "git", "jira", "confluence", "excel", "powerpoint", "word",
  "linux", "unix", "bash", "shell", "sql", "postgresql", "postgres",
  "mysql", "mongo", "mongodb", "snowflake", "bigquery", "redshift",
  "spark", "hadoop", "kafka", "airflow", "dbt", "fivetran", "tableau",
  "power bi", "powerbi", "looker", "metabase", "superset",
  "python", "java", "javascript", "typescript", "node", "react", "vue",
  "angular", "next.js", "django", "flask", "fastapi", "spring", "go", "golang",
  "rust", "c++", "c#", ".net", "php", "ruby", "scala", "swift", "kotlin",
  "aws", "azure", "gcp", "cloud", "lambda", "s3", "ec2", "glue", "athena",
  "langchain", "langgraph", "openai", "groq", "pytorch", "tensorflow",
  "keras", "scikit", "sklearn", "pandas", "numpy", "hugging face",
  "transformers", "ollama", "vllm", "prompt", "rag", "vector",
  "figma", "sketch", "adobe", "photoshop", "illustrator", "xd",
  "salesforce", "crm", "hubspot", "google analytics", "ga4", "webflow",
];

const CONCEPT_KEYWORDS = [
  "machine learning", "deep learning", "neural", "reinforcement learning",
  "supervised", "unsupervised", "nlp", "natural language", "computer vision",
  "llm", "agent", "generative", "classification", "regression", "clustering",
  "forecasting", "time series", "statistics", "statistical", "probability",
  "a/b", "ab testing", "experimentation", "hypothesis", "causal",
  "data pipeline", "etl", "elt", "data modeling", "data warehouse", "data lake",
  "mlops", "devops", "ci/cd", "continuous integration", "deployment",
  "monitoring", "observability", "scalability", "performance optimization",
  "agile", "scrum", "kanban", "sprint", "stakeholder", "cross-functional",
  "microservices", "distributed systems", "architecture", "system design",
  "rest", "graphql", "api design", "unit testing", "test automation", "tdd",
  "security", "compliance", "gdpr", "hipaa", "accessibility", "wcag",
  "roadmap", "product strategy", "user research", "wireframe", "prototype",
  "financial modeling", "budget", "risk", "audit", "kpi", "okr", "metric",
];

function normalize(label: string): string {
  return label.toLowerCase().trim();
}

export function classifyGap(label: string): GapGroup {
  const l = normalize(label);
  if (TOOL_KEYWORDS.some((k) => l.includes(k))) return "tool";
  if (CONCEPT_KEYWORDS.some((k) => l.includes(k))) return "concept";
  return "general";
}

/** Sentinel prefix avoids id collisions across re-runs (React keys). */
let idCounter = 0;
function nextId(): string {
  idCounter = (idCounter + 1) % 1_000_000;
  return "gap-" + idCounter;
}

export const GROUP_ORDER: GapGroup[] = ["tool", "concept", "general"];

export const GROUP_LABELS: Record<GapGroup, string> = {
  tool: "Tools and platforms",
  concept: "Concepts and methods",
  general: "General skills",
};

/**
 * Parses the raw scanner output into unique, usable chips.
 * - splits on newlines (scanner emits one bullet per line)
 * - trims bullets markers like "- ", "• ", "1. "
 * - dedupes case-insensitively
 * - drops stray headings ("Missing Skills:") and overlong fragments
 */
export function parseGapSkills(raw: string): GapItem[] {
  const seen = new Set<string>();
  const items: GapItem[] = [];
  for (const line of (raw || "").split(/\r?\n/)) {
    const label = line.replace(/^\s*(?:[-•*·]|\d+[.)])\s*/, "").trim();
    if (!label || label.length < 2 || label.length > 120) continue;
    if (/^(missing|skills?)\s*:?$/i.test(label)) continue;
    const key = normalize(label);
    if (seen.has(key)) continue;
    seen.add(key);
    items.push({ id: nextId(), label, group: classifyGap(label) });
  }
  return items;
}

export interface ComposedNotes {
  notes: string;
  /** true when nothing was typed and no gaps survived → user skipped curation */
  isDefault: boolean;
}

/** Builds the notes string sent to step 2 (fp-changing composition ONLY here). */
export function composeStep2Notes(
  freeText: string,
  items: GapItem[],
  removedLabels: string[]
): ComposedNotes {
  const lines: string[] = [];
  const text = freeText.trim();
  if (text) lines.push(text);

  const kept = items.filter((i) => (i.note || "").trim());
  const bare = items.filter((i) => !(i.note || "").trim());

  for (const item of kept) {
    lines.push(`- ${item.label}: ${(item.note || "").trim()}`);
  }
  for (const item of bare) {
    lines.push(`- ${item.label}`);
  }
  if (removedLabels.length > 0) {
    lines.push(`Do NOT feature: ${removedLabels.join(", ")}.`);
  }
  return { notes: lines.join("\n"), isDefault: text === "" && items.length === 0 && removedLabels.length === 0 };
}
