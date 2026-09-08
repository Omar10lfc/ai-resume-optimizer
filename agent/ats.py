"""ATS keyword engine: extraction, matching, stemming, tiers, and scoring."""
import re
from collections import Counter

from .helpers import _safe_print, _safe_truncate

# ==========================================
# Stopwords & Low-Signal Terms
# ==========================================

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "can", "shall", "not", "no", "nor", "so", "as", "if", "then",
    "than", "too", "very", "just", "about", "above", "after", "again", "all",
    "also", "am", "any", "because", "before", "between", "both", "each", "few",
    "get", "got", "here", "how", "into", "it", "its", "more", "most", "must",
    "my", "new", "now", "only", "other", "our", "out", "over", "own", "same",
    "she", "he", "they", "them", "their", "this", "that", "these", "those",
    "through", "under", "until", "up", "us", "we", "what", "when", "where",
    "which", "while", "who", "whom", "why", "you", "your", "such", "well",
    "work", "working", "ability", "experience", "strong", "including", "within",
    "using", "used", "use", "based", "skills", "role", "team", "etc", "ie",
    "eg", "across", "per", "via", "re", "like", "make", "sure", "good", "great",
    "looking", "join", "help", "need", "want", "know", "take", "come", "see",
    "think", "look", "give", "day", "year", "way", "part", "even", "back",
    "still", "find", "long", "provide", "high", "right", "build", "key",
    "ideal", "candidate", "requirements", "qualifications", "responsibilities",
    "about", "company", "position", "job", "apply", "application", "will",
    "plus", "bonus", "preferred", "required", "minimum", "years",
    "seeking", "talented", "responsible", "applying", "latest", "practical",
    "technical", "techniques", "libraries", "possess", "necessarily", "below",
    "equivalent", "backed", "proficiency", "proven", "record", "demonstrated",
    "exercise", "judgment", "solving", "complex", "challenges", "function",
    "influential", "member", "highly", "integrated", "composed", "hands-on",
    "non-technical", "equal", "opportunity", "employer", "value", "ethics",
    "dedication", "sustainability", "safeguarding", "respect", "inclusion",
    "interested", "dedicated", "impactful", "projects", "welcome", "applications",
    "qualified", "candidates", "regardless", "background", "areas", "subject",
    "industrial", "fluency", "modern", "environments", "production", "capable",
    "passionate", "driven", "impact", "enterprise", "multiple", "domains",
    "research", "development", "degree", "stem", "master", "successful",
    "many", "maintain", "maintaining", "supporting", "deploying",
    "hands", "proven", "track", "record", "prefer", "preferably",
    "adopting", "ensure", "ensuring", "helping", "includingbut", "etc.",
    "you'll", "we're", "joinus", "hiring", "applynow",
    "implement", "implementing", "contribute", "contributing",
    "collaborate", "collaborating", "enhance", "enhancing",
    "provide", "providing", "configure", "configuring",
    "optimize", "optimizing", "intelligent", "stay", "staying",
    "familiarity", "understanding", "knowledge", "foundation",
    "concepts", "trends", "productivity", "tooling", "documentation",
    "major", "graduation", "grads", "emerging", "current", "senior",
}

_LOW_SIGNAL_TERMS = {
    "support", "engineering", "teams", "computer", "coding", "testing",
    "frameworks", "automation", "workflows", "platforms", "systems",
    "solutions", "services", "processes", "tools", "technologies",
    "assistants", "pipelines", "code", "design", "engineers",
    "integrations", "configurations", "environments", "ides",
}


def _extract_company_names(text: str) -> set[str]:
    names = set()
    patterns = [
        r'(?:[Jj]oin|[Aa]t|[Aa]bout)\s+([A-Z][a-zA-Z]+)',
        r'([A-Z][a-zA-Z]+)\s+is an? ',
        r'([A-Z][a-zA-Z]+)\s+(?:values?|welcomes?|offers?)',
    ]
    for pat in patterns:
        for match in re.findall(pat, text):
            if match.lower() not in _STOPWORDS and len(match) >= 3:
                names.add(match.lower())
    return names


# ==========================================
# Compound Phrases & Short Tech Terms
# ==========================================

_COMPOUND_PHRASES = [
    "machine learning", "deep learning", "natural language processing",
    "computer vision", "data mining", "data science", "data engineering",
    "reinforcement learning", "transfer learning", "generative ai",
    "large language model", "large language models",
    "neural network", "neural networks", "gradient descent",
    "hyperparameter tuning", "model training", "model inference",
    "model serving", "model deployment", "model monitoring",
    "feature engineering", "feature store", "prompt engineering",
    "fine tuning", "fine-tuning", "retrieval augmented generation",
    "vector database", "embedding model", "attention mechanism",
    "support vector machine",
    "scikit-learn", "scikit learn", "hugging face", "huggingface",
    "data pipeline", "data pipelines", "data warehouse", "data lake",
    "etl pipeline", "stream processing", "batch processing",
    "distributed computing", "distributed systems",
    "cloud-native", "cloud native", "infrastructure as code",
    "container orchestration",
    "ci/cd", "ci cd", "continuous integration", "continuous deployment",
    "full stack", "full-stack", "front end", "front-end",
    "back end", "back-end", "rest api", "restful api",
    "a/b testing", "version control", "code review",
    "test driven development", "agile methodology", "design patterns",
    "microservices architecture", "event driven",
    "vector search", "semantic search", "information retrieval",
]

_SHORT_TECH_TERMS = {
    "ai", "ml", "nlp", "cv", "dl", "aws", "gcp", "sql", "api",
    "r", "go", "c#", "c++", "js", "ts", "ui", "ux", "rag", "llm",
    "etl", "s3", "ec2", "rds", "sqs", "sns", "ecs", "eks",
    "node", "node.js",
}

# Version-qualified terms: "python3.11" -> "python", "node20" -> "node", "react18" -> "react"
_VERSIONED_TERMS = {
    "python3": "python", "python2": "python", "python3.11": "python",
    "python3.12": "python", "python3.10": "python", "python3.9": "python",
    "node20": "node", "node18": "node", "node22": "node",
    "react18": "react", "react17": "react", "react19": "react",
    "angular16": "angular", "angular17": "angular", "angular18": "angular",
    "vue3": "vue", "vue2": "vue",
    "dotnet8": "dotnet", "dotnet7": "dotnet", "dotnet6": "dotnet",
    "java21": "java", "java17": "java", "java11": "java",
}

_SYNONYMS = {
    "k8s": ["kubernetes"], "kubernetes": ["k8s"],
    "aws": ["amazon web services"], "amazon web services": ["aws"],
    "gcp": ["google cloud platform", "google cloud"], "google cloud platform": ["gcp"],
    "azure": ["microsoft azure"], "microsoft azure": ["azure"],
    "nlp": ["natural language processing"], "natural language processing": ["nlp"],
    "cv": ["computer vision"], "computer vision": ["cv"],
    "ml": ["machine learning"], "machine learning": ["ml"],
    "dl": ["deep learning"], "deep learning": ["dl"],
    "llm": ["large language model", "large language models"],
    "large language model": ["llm"], "large language models": ["llm"],
    "rag": ["retrieval augmented generation"], "retrieval augmented generation": ["rag"],
    "svm": ["support vector machine"], "support vector machine": ["svm"],
    "cnn": ["convolutional neural network"], "convolutional neural network": ["cnn"],
    "rnn": ["recurrent neural network"], "recurrent neural network": ["rnn"],
    "gan": ["generative adversarial network"], "generative adversarial network": ["gan"],
    "tf": ["tensorflow"], "tensorflow": ["tf"],
    "pt": ["pytorch"], "pytorch": ["pt"],
    "sklearn": ["scikit-learn", "scikit learn"], "scikit-learn": ["sklearn"],
    "hf": ["hugging face", "huggingface"], "hugging face": ["hf", "huggingface"],
    "ci/cd": ["continuous integration", "continuous deployment", "ci cd"],
    "continuous integration": ["ci/cd"], "ci cd": ["ci/cd"],
    "iac": ["infrastructure as code"], "infrastructure as code": ["iac", "terraform"],
    "docker": ["containerization"], "containerization": ["docker"],
    "js": ["javascript"], "javascript": ["js"],
    "ts": ["typescript"], "typescript": ["ts"],
    "python3": ["python"], "python": ["python3"],
    "go": ["golang"], "golang": ["go"],
    "etl": ["extract transform load"],
    "sql": ["structured query language"],
    "postgres": ["postgresql", "postgre sql"],
    "postgresql": ["postgres"],
    "nosql": ["non-relational database", "mongodb", "dynamodb"],
}


# ==========================================
# Stemmer
# ==========================================

def _simple_stem(word: str) -> str:
    """Lightweight suffix stripper for common English technical morphology."""
    w = word.lower()
    suffixes = [
        ("ization", 4), ("isation", 4), ("izing", 4), ("ising", 4),
        ("ments", 4), ("ment", 4), ("ying", 1), ("ting", 4), ("ning", 4),
        ("ing", 3), ("ness", 4), ("tion", 3), ("sion", 3), ("ious", 4),
        ("ical", 4), ("able", 4), ("ible", 4), ("ful", 3), ("ive", 3),
        ("ous", 3), ("ies", 3), ("ers", 3), ("ors", 3), ("ity", 3),
        ("ly", 2), ("ed", 2), ("es", 2), ("er", 2), ("s", 1),
    ]
    for suffix, min_stem_len in suffixes:
        if w.endswith(suffix) and len(w) - len(suffix) >= min_stem_len:
            stem = w[:-len(suffix)]
            if suffix == "ying":
                return stem + "y"
            return stem
    return w


# ==========================================
# Keyword Tiers
# ==========================================

_TIER1_TOOLS = {
    "python", "java", "scala", "rust", "go", "r", "julia",
    "javascript", "typescript", "c++", "c#", "sql",
    "pytorch", "tensorflow", "keras", "jax", "scikit-learn", "sklearn",
    "hugging face", "huggingface", "langchain", "llamaindex", "langgraph",
    "xgboost", "lightgbm", "catboost", "optuna",
    "aws", "gcp", "azure", "sagemaker", "lambda", "ec2", "s3",
    "spark", "pyspark", "hadoop", "kafka", "airflow", "dbt", "snowflake",
    "bigquery", "redshift", "postgresql", "mongodb", "redis", "elasticsearch",
    "docker", "kubernetes", "k8s", "terraform", "ansible",
    "jenkins", "github actions", "gitlab ci",
    "mlflow", "wandb", "dvc", "kubeflow", "bentoml", "triton",
    "faiss", "pinecone", "weaviate", "chromadb", "qdrant",
    "whisper", "bert", "gpt", "llama", "mistral",
    "tableau", "grafana", "prometheus",
    "fastapi", "flask", "django", "streamlit", "gradio",
}

_TIER2_CONCEPTS = {
    "machine learning", "deep learning", "natural language processing",
    "computer vision", "reinforcement learning", "transfer learning",
    "generative ai", "large language model", "neural network",
    "ci/cd", "continuous integration", "mlops", "devops",
    "data pipeline", "data engineering", "feature engineering",
    "model deployment", "model monitoring", "model serving",
    "prompt engineering", "fine tuning", "fine-tuning",
    "retrieval augmented generation", "rag",
    "distributed computing", "microservices",
    "rest api", "restful api", "api",
    "etl", "data warehouse", "data lake",
    "vector database", "semantic search",
    "infrastructure as code", "containerization",
    "agile", "scrum",
}


def _classify_keyword_tier(keyword: str) -> int:
    kw = keyword.lower()
    if kw in _TIER1_TOOLS:
        return 3
    if kw in _TIER2_CONCEPTS:
        return 2
    return 1


# ==========================================
# Matching Helpers
# ==========================================

def _normalize_phrase_key(phrase: str) -> str:
    return re.sub(r'[\s/\-_]+', '', phrase.lower())


def freq_ok(tokens: list[str], term: str, min_count: int = 4) -> bool:
    return tokens.count(term) >= min_count


def _boundary_contains(keyword: str, text_lower: str) -> bool:
    if " " in keyword:
        return keyword in text_lower
    pattern = r'(?<![a-z0-9])' + re.escape(keyword) + r'(?![a-z0-9])'
    return re.search(pattern, text_lower) is not None


def _keyword_found_in_text(keyword: str, text_lower: str) -> tuple[bool, str]:
    if _boundary_contains(keyword, text_lower):
        return True, "exact"
    synonyms = _SYNONYMS.get(keyword, [])
    for syn in synonyms:
        if _boundary_contains(syn, text_lower):
            return True, "synonym"
    if ' ' not in keyword and '/' not in keyword and '-' not in keyword:
        kw_stem = _simple_stem(keyword)
        if len(kw_stem) >= 3:
            words_in_text = re.findall(r'[a-zA-Z#+/.:_-]{2,}', text_lower)
            for word in words_in_text:
                if _simple_stem(word) == kw_stem and word != keyword:
                    return True, "stem"
    return False, ""


def _suggest_section(keyword: str) -> str:
    kw = keyword.lower()
    if kw in _TIER1_TOOLS:
        return "Skills"
    if kw in _TIER2_CONCEPTS:
        if any(x in kw for x in ["pipeline", "deployment", "monitoring", "serving"]):
            return "Experience or Projects"
        return "Skills or Summary"
    if any(x in kw for x in ["cloud", "aws", "gcp", "azure", "docker", "kubernetes",
                               "infrastructure", "ci/cd", "terraform"]):
        return "Cloud & MLOps skills"
    return "Skills or Experience"


# ==========================================
# Keyword Extraction
# ==========================================

def _extract_keywords(text: str, top_n: int = 20) -> list[str]:
    text_lower = text.lower()
    company_names = _extract_company_names(text)

    found_compounds = []
    seen_phrases = set()
    for phrase in _COMPOUND_PHRASES:
        if phrase in text_lower and _normalize_phrase_key(phrase) not in seen_phrases:
            found_compounds.append(phrase)
            seen_phrases.add(_normalize_phrase_key(phrase))

    compound_words = set()
    for phrase in found_compounds:
        compound_words.update(re.findall(r'[a-z]+', phrase))

    tokens = re.findall(r'[a-zA-Z#+/.:-]{2,}', text_lower)
    tokens = [t.strip('.-/:') for t in tokens if t.strip('.-/:')]

    # Detect version-qualified terms (python3.11, node20, react18) and map to base form
    versioned_keywords = []
    for token in tokens:
        if token in _VERSIONED_TERMS:
            base = _VERSIONED_TERMS[token]
            if base not in versioned_keywords:
                versioned_keywords.append(base)

    _compound_keys = {_normalize_phrase_key(p) for p in _COMPOUND_PHRASES}

    meaningful = [
        t for t in tokens
        if t not in _STOPWORDS
        and t not in compound_words
        and t not in company_names
        and _normalize_phrase_key(t) not in seen_phrases
        and ("-" not in t or t in _SHORT_TECH_TERMS
             or _normalize_phrase_key(t) in _compound_keys)
        and (len(t) >= 3 or t in _SHORT_TECH_TERMS)
        and (t not in _LOW_SIGNAL_TERMS or freq_ok(tokens, t))
    ]

    freq = Counter(meaningful)
    single_keywords = [kw for kw, _ in freq.most_common(top_n * 2)]

    combined = found_compounds + versioned_keywords + single_keywords
    return combined[:top_n]


# ==========================================
# Job Brief (for fast LLM calls)
# ==========================================

def _job_brief(job_text: str, max_chars: int = 400) -> str:
    keywords = _extract_keywords(job_text, top_n=12)
    return (
        f"Role context: {_safe_truncate(job_text, max_chars, 'Job context')}\n"
        f"Key requirements/keywords: {', '.join(keywords)}"
    )


# ==========================================
# Main ATS Scoring
# ==========================================

def compute_ats_match(job_text: str, resume_text: str,
                      semantic_matches: set[str] | None = None) -> dict:
    semantic_matches = semantic_matches or set()
    job_keywords = _extract_keywords(job_text, top_n=20)
    resume_lower = resume_text.lower()

    matched_stems = set()
    for kw in job_keywords:
        if len(re.findall(r'[a-z]+', kw)) > 1:
            for word in re.findall(r'[a-z]+', kw):
                matched_stems.add(_simple_stem(word))

    results = []
    seen_stems = set()

    for kw in job_keywords:
        if len(re.findall(r'[a-z]+', kw)) == 1:
            stem = _simple_stem(kw)
            if stem in seen_stems:
                continue
            seen_stems.add(stem)

        found, method = _keyword_found_in_text(kw, resume_lower)
        if not found and kw.lower() in semantic_matches:
            found, method = True, "semantic"
        tier = _classify_keyword_tier(kw)
        tier_label = {3: "Tool", 2: "Concept", 1: "General"}[tier]

        results.append({
            "keyword": kw,
            "found": found,
            "method": method,
            "tier": tier,
            "tier_label": tier_label,
            "weight": tier,
            "status": "+" if found else "-"
        })

    total_weight = sum(r["weight"] for r in results)
    earned_weight = sum(r["weight"] for r in results if r["found"])
    percentage = round((earned_weight / total_weight * 100), 1) if total_weight > 0 else 0

    match_count = sum(1 for r in results if r["found"])
    total = len(results)

    lines = [f"## ATS Keyword Match: {percentage}% (weighted)"]
    lines.append(f"*{match_count} of {total} keywords found - weighted by importance*")
    lines.append("")

    found_keywords = sorted(
        [r for r in results if r["found"]],
        key=lambda r: r["tier"], reverse=True
    )
    missing_keywords = sorted(
        [r for r in results if not r["found"]],
        key=lambda r: r["tier"], reverse=True
    )

    if found_keywords:
        lines.append("### Found in Resume")
        for tier, tier_name in ((3, "Tools"), (2, "Concepts"), (1, "General")):
            kws = [r["keyword"] for r in found_keywords if r["tier"] == tier]
            if kws:
                lines.append(f"- **{tier_name}:** {', '.join(f'`{k}`' for k in kws)}")
        semantic_count = sum(1 for r in found_keywords if r["method"] == "semantic")
        if semantic_count:
            lines.append(f"*{semantic_count} keyword(s) matched semantically - your resume expresses them in different words.*")
        lines.append("")

    if missing_keywords:
        lines.append("### Missing from Resume")
        for r in missing_keywords:
            suggestion = _suggest_section(r["keyword"])
            lines.append(f"- `{r['keyword']}` - *try adding to: {suggestion}*")
        lines.append("")

    if percentage >= 80:
        lines.append("> **Strong match** - your resume covers most key terms.")
    elif percentage >= 50:
        lines.append("> **Moderate match** - consider adding the missing keywords where truthful.")
    else:
        lines.append("> **Low match** - significant keyword gaps. Review the missing terms carefully.")

    lines.append("")
    lines.append("---")
    lines.append(f"*Scoring: {earned_weight}/{total_weight} weighted points "
                 f"(tools x3, concepts x2, general x1)*")

    return {
        "keywords": results,
        "match_count": match_count,
        "total": total,
        "percentage": percentage,
        "weighted_earned": earned_weight,
        "weighted_total": total_weight,
        "formatted": "\n".join(lines)
    }
