"""Step 2 layout parity, wrapping, parser, and page-target tests."""
from pathlib import Path
import tempfile
from unittest.mock import patch

from hypothesis import HealthCheck, given, seed, settings, strategies as st
from hypothesis.database import DirectoryBasedExampleDatabase
import pdfplumber
import pytest

from agent import config
from agent.html_render import HAVE_BROWSER, html_to_pdf, render_density_tiers, render_resume_html
from agent.latex_render import (
    HAVE_PDFLATEX,
    latex_to_pdf,
    parse_markdown_resume,
    pdf_page_count,
    render_compact_tiers,
    render_resume_tex,
)
from agent.nodes.loader import loader_node
from agent.nodes.pdf_exporter import export_documents
from agent.pdf_verify import verify_pdf


# 13 diverse archetypes matching the audit corpus (1-page & 2-page, non-ASCII,
# long company/project titles, 0-3 links, stacked multi-degree Education,
# reordered sections, special symbols, non-tech role, dense technical profile).
CORPUS_13 = [
    (
        "01_junior_swe",
        1,
        """# Maya Lin
Seattle, WA | 206-555-0142 | maya.lin@example.edu | [GitHub](https://github.com/mayalin) | [LinkedIn](https://linkedin.com/in/mayalin)

## Summary
Computer Science graduate with backend API, SQL, and automated testing experience building containerized web services.

## Education
University of Washington, Seattle, WA
B.S. in Computer Science, Jun 2025

## Experience
Software Engineering Intern | CloudScale Systems | Jun 2024 -- Sep 2024
- Built FastAPI microservices backed by PostgreSQL handling 1,200 req/s and cutting p95 latency from 240ms to 110ms.
- Wrote 85 pytest integration tests in GitHub Actions CI, raising branch coverage from 61% to 88%.

## Projects
TaskQueue Engine | [GitHub](https://github.com/mayalin/tq) | Python, Redis, Docker | Jan 2025 -- Mar 2025
- Implemented priority job scheduling with Redis streams and dead-letter retries reducing dropped tasks to 0%.

## Technical Skills
**Languages**: Python, TypeScript, SQL, Go
**Backend**: FastAPI, PostgreSQL, Redis, Docker, GitHub Actions
""",
    ),
    (
        "02_senior_staff_dense",
        1,
        """# Marcus Vance
San Francisco, CA | marcus@vance.dev | [GitHub](https://github.com/mvance)

## Summary
Staff Distributed Systems Engineer with 9+ years scaling event-driven payment ledgers and multi-region Kubernetes compute clusters.

## Experience
Staff Software Engineer | Meridian Payments | Mar 2021 -- Present
- Architected multi-region Kafka payment settlement backbone processing 45,000 TPS at 99.995% availability across 3 AWS regions.
- Eliminated $1.4M/yr cross-AZ egress cost by redesigning consumer group locality and protobuf payload compression.

Senior Software Engineer | Harbor Cloud Inc. | Jun 2018 -- Feb 2021
- Led migration from monolith to 18 Go gRPC services with zero customer-facing downtime across 600 enterprise tenants.
- Reduced Terraform deployment rollback frequency from 11% to 1.5% by introducing progressive canary health gates.

## Education
University of California, Berkeley, CA
B.S. in Electrical Engineering and Computer Sciences, May 2016

## Technical Skills
**Systems**: Go, Rust, Python, Kafka, PostgreSQL, Kubernetes, Terraform, AWS
""",
    ),
    (
        "03_long_project_title_3links",
        1,
        """# Nadia Al-Mansoor
Cairo, Egypt | nadia.mansoor@example.org | [GitHub](https://github.com/nadia)

## Projects
Enterprise Multi-Tenant Autonomous Retrieval and Real-Time Compliance Auditing Platform for Regulated Healthcare | [Live Demo](https://demo.example.com/health) | [GitHub](https://github.com/nadia/health-audit) | [Paper](https://arxiv.org/abs/2501.00001) | Python, LangGraph, FAISS | Jan 2025 -- May 2025
- Cut policy verification turnaround by 64% using hybrid BM25 + cross-encoder retrieval over 120,000 regulatory clauses.
- Enforced zero-hallucination citation grounding across 99.2% of sampled audit responses.

CLI Tool | Python | Feb 2024 -- Mar 2024
- Built CSV validator used by 15 analysts.

## Experience
AI Engineer | Nile Data Systems | Jun 2024 -- Present
- Deployed FastAPI inference gateway serving 8,000 daily compliance queries within a 450ms p95 budget.

## Education
Cairo University, Giza, Egypt
B.Sc. in Computer Engineering, Jul 2024
""",
    ),
    (
        "04_zero_link_minimal",
        1,
        """# Jordan Rivera
Austin, TX | jordan.rivera@example.com

## Experience
Operations Analyst | LoneStar Logistics | Jan 2023 -- Present
- Consolidated weekly warehouse exception tracking into automated SQL reports saving 14 analyst hours per week.
- Negotiated carrier SLAs reducing damaged freight claims by 22% across 18 regional hubs.

## Education
Texas State University, San Marcos, TX
B.B.A. in Supply Chain Management, Dec 2022

## Skills
**Operations**: SQL, Excel, Tableau, ERP Auditing, Freight Routing
""",
    ),
    (
        "05_long_company_exp",
        1,
        """# Priya Nair
Boston, MA | priya.nair@example.com | [LinkedIn](https://linkedin.com/in/priyanair)

## Experience
Lead Principal Enterprise Cloud Modernization & Security Architect | International Federated Telecommunications & Aerospace Holdings Corporation Group Ltd. | Jan 2022 -- Present
- Directed 22 engineers migrating 140 legacy mainframe billing workflows to containerized microservices on AWS EKS.
- Automated SOC2 continuous compliance checks reducing external audit evidence collection from 6 weeks to 4 days.

## Education
Georgia Institute of Technology, Atlanta, GA
M.S. in Computer Science, Dec 2021

## Technical Skills
**Cloud**: AWS EKS, Terraform, Kubernetes, Python, Go, SOC2, OpenTelemetry
""",
    ),
    (
        "06_non_ascii_intl",
        1,
        """# \u00c5se Lindstr\u00f6m-M\u00fcller
Z\u00fcrich, Switzerland | ase.muller@example.ch | [GitHub](https://github.com/asemuller)

## Summary
Data Scientist specializing in multilingual NLP and demand forecasting across S\u00e3o Paulo, Montr\u00e9al, and M\u00fcnchen retail markets.

## Experience
Senior Data Scientist | Helvetia Analytics GmbH | Sep 2021 -- Present
- Built hierarchical LightGBM & Prophet demand forecaster cutting perishable waste by 18.4% across 420 Swiss supermarkets.
- Calibrated multilingual transformer embeddings achieving 0.89 F1 on German, French, and Portuguese product taxonomy mapping.

## Education
ETH Z\u00fcrich, Z\u00fcrich, Switzerland
M.Sc. in Data Science, Jun 2021

## Technical Skills
**ML & Languages**: Python, LightGBM, PyTorch, SQL, Docker
""",
    ),
    (
        "07_sparse_bullets",
        1,
        """# Kevin Brooks
Denver, CO | kevin.brooks@example.net

## Summary
QA Automation Engineer focused on Playwright end-to-end suites and API contract verification.

## Experience
QA Engineer | Peak Commerce | May 2022 -- Present
- Maintained end-to-end Playwright test suite covering checkout and refund flows across Chrome and Firefox.

## Education
University of Colorado Boulder, Boulder, CO
B.S. in Information Science, May 2022
""",
    ),
    (
        "08_one_page_overflow_candidate",
        1,
        """# Omar Hassan
Alexandria, Egypt | +20-100-555-0199 | omar.hassan@example.org | [GitHub](https://github.com/ohassan) | [LinkedIn](https://linkedin.com/in/ohassan)

## Summary
Machine Learning & Backend Engineer building hybrid RAG search pipelines, OR-Tools optimization engines, and evaluated FastAPI services.

## Experience
ML Engineer Intern | Delta Port Systems | Jul 2024 -- Sep 2024
- Built CP-SAT berth allocation optimizer slashing average vessel wait time 41% across 900 scheduled arrivals within 5s timeout.
- Implemented automated drift alerts with IsolationForest catching sensor faults 36 hours ahead of manual operator review.
- Added 70 pytest integration tests covering API schema validation and fallback retry paths.

Computer Vision Trainee | National Tech Institute | Aug 2023 -- Sep 2023
- Trained YOLOv11 detector reaching 94.1% mAP@50 at 1.9ms latency across 2,100 annotated validation frames.
- Benchmarked ResNet50 and ConvNeXt baselines on aerial harbor imagery lifting defect recall by 11 points.

## Projects
Hybrid Arabic QA Engine | [Live Demo](https://demo.example.com/qa) | [GitHub](https://github.com/ohassan/arqa) | Python, FAISS, BM25, FastAPI | Jan 2025 -- May 2025
- Improved top-5 retrieval MRR by 9% over dense-only baseline using weighted reciprocal rank fusion + cross-encoder reranking.
- Lowered median LLM inference cost 65% via two-tier model routing with rate-limit fallback.

Speech Summarizer | [GitHub](https://github.com/ohassan/speech) | Whisper, AraBART, Gradio | Sep 2024 -- Dec 2024
- Reduced Arabic ASR WER from 41.2 to 21.8 via fine-tuning Whisper-small and publishing reproducible Hugging Face checkpoints.

## Education
Egypt-Japan University of Science and Technology, Alexandria, Egypt
B.Sc. in Computer Science and AI, Jun 2025

## Technical Skills
**AI & ML**: PyTorch, Transformers, FAISS, Whisper, OR-Tools, scikit-learn
**Backend & Cloud**: Python, FastAPI, Docker, PostgreSQL, GitHub Actions, AWS
""",
    ),
    (
        "09_non_tech_finance",
        1,
        """# Claire Dubois
New York, NY | claire.dubois@example.com | [LinkedIn](https://linkedin.com/in/clairedubois)

## Summary
Corporate FP&A Director with 8 years leading $650M P&L budgeting, M&A valuation modeling, and executive treasury reporting.

## Experience
Director of FP&A | Hudson Consumer Brands | Feb 2022 -- Present
- Led $650M annual operating budget cycle across 9 business units improving quarterly EBITDA forecast accuracy within 1.8%.
- Built 13-week rolling cash-flow model in Anaplan freeing $28M in trapped working capital.

Senior Financial Analyst | Meridian Capital Partners | Jun 2018 -- Jan 2022
- Built leveraged buyout and DCF valuation models supporting $410M in closed mid-market industrial acquisitions.

## Education
New York University Stern School of Business, New York, NY
M.B.A. in Finance & Accounting, May 2018

## Skills
**Finance**: FP&A, DCF/LBO Modeling, GAAP, Treasury, Anaplan, SAP S/4HANA, SQL
""",
    ),
    (
        "10_reordered_edu_long",
        1,
        """# Tobias Lindqvist
Stockholm, Sweden | tobias.l@example.se | [GitHub](https://github.com/tobiasl)

## Technical Skills
**Core**: Go, Rust, Kubernetes, gRPC, PostgreSQL, Terraform

## Projects
MeshProxy | [Source](https://github.com/tobiasl/meshproxy) | Go, eBPF | Mar 2025 -- May 2025
- Intercepted pod-to-pod TCP sockets via eBPF sockops reducing sidecar CPU consumption by 29%.

## Experience
Systems Engineer | NordicGrid AB | Jan 2023 -- Present
- Operated 300-node bare-metal Kubernetes cluster with zero unplanned downtime over 18 months.

## Honors & Awards
- Nordic Distributed Systems Hackathon Winner | Royal Institute Hackathon Committee | Nov 2024

## Education
KTH Royal Institute of Technology & Technical University of Munich (Joint Track), Stockholm & Munich
M.Sc. in Distributed Computing & Cloud Systems Infrastructure, Jun 2023
""",
    ),
    (
        "11_special_chars",
        1,
        """# Raj Patel
Toronto, Canada | raj.patel@example.ca | [GitHub](https://github.com/rajpatel)

## Summary
Quant & Backend Engineer building C++20/Python pricing engines, R&D backtesters, and S&P 500 execution analytics at 99.9% uptime.

## Experience
Quant Developer | MapleLeaf Capital (R&D Division) | May 2022 -- Present
- Engineered C++ & Python option-pricing service computing ~100% Greeks across $4.2B notional books with <0.01% error.
- Reduced tail latency by 42% (180us -> 104us) via lock-free ring buffers and cache-aligned structs.

## Education
University of Waterloo, Waterloo, Canada
B.Math in Computational Finance & CS, Apr 2022

## Technical Skills
**Languages & Tech**: C++20, C#, Python, F&O Analytics, SQL, R&D Tooling, S&P_500 Feeds
""",
    ),
    (
        "12_two_page_cv",
        2,
        """# Dr. Elena Rostova
London, UK | +44-20-7946-0192 | e.rostova@example.ac.uk | [GitHub](https://github.com/erostova) | [LinkedIn](https://linkedin.com/in/erostova)

## Summary
Principal Research Scientist with 10 years leading NLP, speech recognition, and multimodal foundation model teams from academic research through 10M+ user production deployment.

## Experience
Principal Research Scientist | Albion Conversational AI | Jun 2022 -- Present
- Directed 14-researcher speech & dialogue group shipping multilingual ASR across 28 languages reducing aggregate WER by 31%.
- Designed reinforcement-learning preference optimization pipeline cutting hallucination rates by 44% on internal safety evals.
- Secured $6.2M Horizon Europe research grant and authored 4 peer-reviewed ACL/EMNLP papers.
- Mentored 6 PhD interns to full-time offers and instituted reproducible experiment tracking across 64 A100 GPUs.
- Built streaming ONNX decoder service handling 12,000 concurrent WebSocket audio sessions across EU-West clusters.

Senior Applied Scientist | Thames Search Engine Ltd | Sep 2019 -- May 2022
- Built dense passage retrieval index scaling to 180M documents at 22ms p95 query latency.
- Replaced rule-based query rewriting with distilled T5 seq2seq model lifting zero-result recovery by 27%.
- Co-designed automated human evaluation guidelines across 120 annotators with 0.84 Cohen's kappa.
- Orchestrated multi-node FAISS shard rebalancing cutting cold-failover recovery time from 14 minutes to 38 seconds.

Applied Scientist | CamBridge Speech Labs | Jul 2016 -- Aug 2019
- Developed CTC-attention hybrid acoustic models improving noisy telephony WER from 19.4% to 12.1%.
- Shipped C++ quantized inference SDK deployed on 4M embedded automotive head units.
- Reduced model memory footprint by 58% through mixed-precision INT8 quantization-aware training.

Research Associate | Imperial College London NLP Group | Sep 2014 -- Jun 2016
- Investigated cross-lingual transfer in dependency parsing across 16 Slavic and Romance languages.
- Co-authored open-source morphology toolkit downloaded 18,000+ times across academic labs.

Graduate Research Assistant | University of Edinburgh | Sep 2012 -- Aug 2014
- Evaluated Bayesian hidden Markov taggers on low-resource Celtic and Finno-Ugric news corpora across 4 benchmark shared tasks.
- Built distributed corpus cleaning pipeline processing 450M web tokens on university compute clusters.
- Designed active-learning annotation UI cutting linguistic treebank labeling hours by 38% across 18 student annotators.

## Projects
OpenSpeechBench | [GitHub](https://github.com/erostova/osb) | [Paper](https://aclanthology.org/example) | PyTorch, Kaldi | Jan 2023 -- Dec 2023
- Published standardized streaming ASR robustness benchmark cited 340+ times across academia and industry labs.
- Reproduced 14 acoustic noise corruption suites and open-sourced baseline Conformer checkpoints under Apache 2.0 license.

LexiAlign | [GitHub](https://github.com/erostova/lexialign) | Transformers, FAISS | Mar 2021 -- Nov 2021
- Open-sourced bilingual sentence aligner processing 50M parallel pairs in under 4 hours on 4 GPUs.
- Validated bitext filtering thresholds on WMT20 low-resource pairs boosting BLEU by +2.1 across 6 language directions.

AcousticSynth | [GitHub](https://github.com/erostova/acousticsynth) | Python, Torchaudio | Feb 2020 -- Sep 2020
- Generated synthetic room-impulse augmentation library cutting far-field ASR errors by 19%.

MultilingualNER | [GitHub](https://github.com/erostova/mner) | spaCy, BERT | Jan 2019 -- Aug 2019
- Released named-entity tagger covering 14 European languages with quantized ONNX runtime.

CrossLingualEval | [GitHub](https://github.com/erostova/cleval) | PyTorch, Datasets | Jun 2018 -- Dec 2018
- Built unified zero-shot transfer harness benchmarking mBERT and XLM-R across 24 typology-diverse languages.

## Publications
- Rostova et al., "Robust Streaming CTC Decoding under Packet Loss", ACL 2024 | Association for Computational Linguistics | Jul 2024
- Rostova et al., "Distilled Cross-Encoders for Low-Latency Passage Reranking", EMNLP 2022 | ACL Anthology | Dec 2022
- Rostova et al., "Quantized Conformer Acoustic Models for Embedded Automotive Hardware", Interspeech 2021 | ISCA | Sep 2021
- Rostova et al., "Cross-Lingual Morphological Disambiguation via Joint Subword Lattices", EACL 2017 | ACL | Apr 2017

## Education
Imperial College London, London, UK
Ph.D. in Natural Language Processing, Jun 2016
University of Edinburgh, Edinburgh, UK
M.Sc. in Artificial Intelligence (Distinction), Sep 2012
University of Bristol, Bristol, UK
B.Sc. in Computer Science (First Class Honours), Jun 2011

## Technical Skills
**Research & ML**: PyTorch, Transformers, Whisper, Kaldi, FAISS, TensorRT, RLHF/DPO
**Engineering**: Python, C++17, Docker, Kubernetes, Slurm, Ray, Linux
""",
    ),
    (
        "13_dense_ai_skills",
        1,
        """# Samira Farid
Cairo, Egypt | +20-120-555-0188 | samira.farid@example.com | [GitHub](https://github.com/sfarid) | [LinkedIn](https://linkedin.com/in/sfarid)

## Summary
AI Systems Engineer experienced in LangGraph multi-agent orchestration, hybrid RAG evaluation, FastAPI deployment, and AWS CI/CD automation.

## Experience
AI Research Engineer | Pharos Intelligence | May 2024 -- Present
- Built hybrid RAG retrieval service (BM25 + dense vector + cross-encoder reranking) lifting P@5 from 0.64 to 0.87.
- Cut LLM inference cost 68% with dynamic tier routing and automated rate-limit failover across 1,400 daily runs.

Software Engineering Intern | Pyramid Cloud Lab | Jun 2023 -- Sep 2023
- Implemented asynchronous FastAPI batch workers backed by Redis and PostgreSQL with 92% pytest coverage.

## Projects
Autonomous Port Scheduler | [Live Demo](https://demo.example.com/port) | [GitHub](https://github.com/sfarid/port) | Python, OR-Tools | Jan 2025 -- May 2025
- Solved berth-allocation optimization via CP-SAT cutting wait times 44% on 1,500 simulated vessel arrivals.

## Education
Cairo University, Giza, Egypt
B.Sc. in Computer Engineering, Jun 2024

## Technical Skills
**Optimization & Operations Research**: OR-Tools CP-SAT, Combinatorial Optimization, IsolationForest, Anomaly Detection
**LLM & Generative AI**: Retrieval-Augmented Generation (RAG), Multi-Agent Systems, LLM Orchestration, Prompt Engineering, LangGraph
**Cloud & MLOps**: AWS, Infrastructure as Code, REST API Development, Model Deployment, Test Automation, CI/CD pipeline setup, automated testing, infrastructure provisioning
""",
    ),
]


def _extract_heading_order(pdf_path: str, expected_headings: list[str]) -> list[str]:
    with pdfplumber.open(pdf_path) as doc:
        full_text = "\n".join(pg.extract_text() or "" for pg in doc.pages)
    norm = full_text.lower()
    positions = []
    for h in expected_headings:
        idx = norm.find(h.lower())
        assert idx >= 0, f"missing heading {h!r} in {pdf_path}"
        positions.append((idx, h.lower()))
    positions.sort()
    return [h for _, h in positions]


@pytest.mark.pdf_render
@pytest.mark.parametrize("name,target_pages,md", CORPUS_13, ids=[c[0] for c in CORPUS_13])
def test_13_resume_latex_vs_html_classic_parity(name, target_pages, md, tmp_path):
    """Both LaTeX and HTML classic compile, pass verify_pdf geometry, and match structure/pages."""
    if not HAVE_PDFLATEX or not HAVE_BROWSER:
        pytest.skip("pdflatex and Chromium browser are both required for cross-engine parity")

    parsed = parse_markdown_resume(md)
    expected_headings = [s.title for s in parsed.sections if s.entries or s.simple_lines]

    # Render LaTeX: compact tiers if target_pages == 1, base for 2-page CV if needed
    tex_base = render_resume_tex(parsed, compact=False)
    l_pdf = latex_to_pdf(tex_base, tmp_path, f"{name}_latex.pdf")
    if (l_pdf is None or (pdf_page_count(l_pdf) or 99) > target_pages) and target_pages == 1:
        compact_res = render_compact_tiers(parsed, tmp_path, f"{name}_latex.pdf", target_pages=target_pages)
        assert compact_res is not None, f"{name}: LaTeX compact tiers failed"
        l_pdf = compact_res["pdf_path"]
    elif l_pdf is None or (pdf_page_count(l_pdf) or 99) > target_pages:
        compact_res = render_compact_tiers(parsed, tmp_path, f"{name}_latex.pdf", target_pages=target_pages)
        if compact_res:
            l_pdf = compact_res["pdf_path"]
    assert l_pdf is not None, f"{name}: LaTeX failed to produce PDF"

    # Render HTML classic
    html_res = render_density_tiers(parsed, tmp_path, f"{name}_html.pdf", theme="classic", target_pages=target_pages)
    if html_res is None:
        html = render_resume_html(parsed, theme="classic")
        h_pdf = html_to_pdf(html, tmp_path, f"{name}_html.pdf")
    else:
        h_pdf = html_res["pdf_path"]
    assert h_pdf is not None, f"{name}: HTML classic failed to produce PDF"

    lv = verify_pdf(l_pdf, expected_pages=target_pages)
    hv = verify_pdf(h_pdf, expected_pages=target_pages)
    assert lv["ok"], f"{name}: LaTeX verify_pdf failed: {lv['checks']}"
    assert hv["ok"], f"{name}: HTML verify_pdf failed: {hv['checks']}"

    assert lv["pages"] <= target_pages, f"{name}: LaTeX pages={lv['pages']} > {target_pages}"
    assert hv["pages"] <= target_pages, f"{name}: HTML pages={hv['pages']} > {target_pages}"
    assert lv["pages"] == hv["pages"], f"{name}: page mismatch LaTeX={lv['pages']} vs HTML={hv['pages']}"

    l_order = _extract_heading_order(l_pdf, expected_headings)
    h_order = _extract_heading_order(h_pdf, expected_headings)
    assert l_order == h_order, (
        f"{name}: cross-engine section order mismatch LaTeX={l_order} HTML={h_order}"
    )
    assert set(l_order) == {h.lower() for h in expected_headings}


@pytest.mark.pdf_render
def test_long_company_and_stacked_education_and_link_dedup(tmp_path):
    """Regression guard for wrapped company rows, stacked degrees, and project link dedup."""
    # Stacked multi-degree Education stays separate entries
    edu_md = next(md for name, _, md in CORPUS_13 if name == "12_two_page_cv")
    parsed_edu = parse_markdown_resume(edu_md)
    edu_sec = next(s for s in parsed_edu.sections if s.title.lower() == "education")
    assert len(edu_sec.entries) == 3
    assert [e.header for e in edu_sec.entries] == [
        "Imperial College London",
        "University of Edinburgh",
        "University of Bristol",
    ]

    # Distinct labels pointing to distinct URLs never duplicate phantom bare links
    proj_sec = next(s for s in parsed_edu.sections if s.title.lower() == "projects")
    osb = next(e for e in proj_sec.entries if "OpenSpeechBench" in e.header)
    combined_links = osb.sub + " " + osb.right + " " + " ".join(osb.links)
    assert "[GitHub](https://github.com/erostova/osb)" in combined_links
    assert "[Paper](https://aclanthology.org/example)" in combined_links
    tex_osb = render_resume_tex(parsed_edu)
    assert tex_osb.count("https://github.com/erostova/osb") == 1
    assert tex_osb.count("https://aclanthology.org/example") == 1

    if HAVE_PDFLATEX:
        long_co_md = next(md for name, _, md in CORPUS_13 if name == "05_long_company_exp")
        tex = render_resume_tex(parse_markdown_resume(long_co_md))
        assert r"\begin{tabularx}" in tex
        pdf = latex_to_pdf(tex, tmp_path, "long_co.pdf")
        assert pdf is not None
        vr = verify_pdf(pdf, expected_pages=1)
        assert vr["ok"], vr


def test_loader_detects_page_target_and_respects_override(tmp_path):
    """PDF input page count sets auto target_pages (1 or 2), text defaults 1, manual wins."""
    # Pasted text auto -> 1
    out_text = loader_node({"original_resume": "Jane Doe\nExperience\n- Did work", "job_description": "Role", "target_pages": 0})
    assert out_text["target_pages"] == 1
    assert out_text["input_pages"] == 0

    # Manual override -> 2 even on plain text
    out_override = loader_node({"original_resume": "Jane Doe", "job_description": "Role", "target_pages": 2})
    assert out_override["target_pages"] == 2

    # 2-page PDF input auto -> 2, and manual 1 override -> 1
    pdf_2p = tmp_path / "cv_2p.pdf"
    pdf_2p.write_bytes(b"%PDF-1.4\n%stub")
    with patch("agent.nodes.loader._load_resume_pdf", return_value="Jane Doe\nExperience"), \
         patch("agent.latex_render.pdf_page_count", return_value=2):
        out_two = loader_node({"original_resume": str(pdf_2p), "job_description": "Role", "target_pages": 0})
        assert out_two["input_pages"] == 2
        assert out_two["target_pages"] == 2

        out_forced_1 = loader_node({"original_resume": str(pdf_2p), "job_description": "Role", "target_pages": 1})
        assert out_forced_1["input_pages"] == 2
        assert out_forced_1["target_pages"] == 1


def test_export_documents_expands_page_target_only_with_human_notes(tmp_path):
    """+1 page expansion up to cap 2 occurs ONLY when human_notes added content."""
    fake_2p_pdf = tmp_path / "resume_2p.pdf"
    fake_2p_pdf.write_bytes(b"%PDF-1.4\n%stub")

    def fake_latex_export(md, out_dir, theme="", target_pages=1):
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / "resume.pdf"
        p.write_bytes(b"%PDF-1.4\n%stub")
        return {
            "resume_pdf_path": str(p),
            "resume_docx_path": "",
            "resume_tex_path": "",
            "resume_pages": 2,
            "resume_compacted": True,
            "page_overflow_warning": "" if target_pages >= 2 else "Exceeded 1 page",
            "render_engine": "latex@1",
            "layout_findings": [],
        }

    with patch("agent.nodes.pdf_exporter._export_resume", side_effect=fake_latex_export), \
         patch("agent.nodes.pdf_exporter._html_to_pdf", return_value=""), \
         patch("agent.nodes.pdf_exporter.verify_pdf", return_value={"pages": 2, "ok": True, "checks": [
             {"name": "readable", "ok": True, "detail": "2 page(s)"},
             {"name": "geometry", "ok": True, "detail": "ok"},
         ]}):
        # Without human_notes: stays strict 1-page target
        no_notes = export_documents(
            "# Jane Doe\n## Summary\nHello", "", tmp_path / "r1",
            target_pages=1, input_pages=1, has_human_notes=False,
        )
        assert no_notes["target_pages"] == 1
        assert no_notes["page_target_expanded"] is False
        assert no_notes["page_overflow_warning"]

        # With human_notes: expands to 2 pages and records note
        with_notes = export_documents(
            "# Jane Doe\n## Summary\nHello", "", tmp_path / "r2",
            target_pages=1, input_pages=1, has_human_notes=True,
        )
        assert with_notes["target_pages"] == 2
        assert with_notes["page_target_expanded"] is True
        assert "Added content moved this to 2 pages." in with_notes["page_overflow_warning"]
        pt_check = next(c for c in with_notes["pdf_checks"] if c["name"] == "page_target")
        assert "Added content moved this to 2 pages." in pt_check["detail"]


_NAMES = [
    "Maya Lin",
    "Dr. Elena Rostova",
    "Jos\u00e9 \u00c1lvarez-Mu\u00f1oz",
    "\u00c5se Lindstr\u00f6m-M\u00fcller",
]
_COMPANIES = [
    "CloudScale Systems",
    "Helvetia Analytics GmbH",
    "International Federated Telecommunications & Aerospace Holdings Corporation Group Ltd.",
    "NordicGrid AB",
]
_TITLES = [
    "Software Engineer",
    "Lead Principal Enterprise Cloud Modernization & Security Architect",
    "Senior Applied Scientist",
]
_PROJ_TITLES = [
    "CLI Tool",
    "OpenSpeechBench",
    "Enterprise Multi-Tenant Autonomous Retrieval and Real-Time Compliance Auditing Platform",
]
_LINKS = [
    "[Live Demo](https://demo.example.com/app)",
    "[GitHub](https://github.com/example/repo)",
    "[Paper](https://aclanthology.org/2025.acl-1.10)",
]


@st.composite
def _arbitrary_resume_md(draw):
    name = draw(st.sampled_from(_NAMES))
    include_phone = draw(st.booleans())
    contact_parts = ["Z\u00fcrich & Boston", "candidate@example.com"]
    if include_phone:
        contact_parts.append("+1-617-555-0194")
    contact_parts.append("[GitHub](https://github.com/candidate)")

    # Experience entries
    exp_count = draw(st.integers(min_value=1, max_value=2))
    exp_lines = ["## Experience"]
    for _ in range(exp_count):
        role = draw(st.sampled_from(_TITLES))
        co = draw(st.sampled_from(_COMPANIES))
        exp_lines.append(f"{role} | {co} | Jan 2022 -- Present")
        b_count = draw(st.integers(min_value=1, max_value=2))
        for bi in range(b_count):
            exp_lines.append(
                f"- Engineered Python/Go microservice pipeline #{bi + 1} reducing p95 latency by {18 + bi * 7}% across 40k daily requests."
            )

    # Project entries with 0..3 links
    proj_count = draw(st.integers(min_value=1, max_value=2))
    proj_lines = ["## Projects"]
    for _ in range(proj_count):
        ptitle = draw(st.sampled_from(_PROJ_TITLES))
        n_links = draw(st.integers(min_value=0, max_value=3))
        link_str = (" | " + " | ".join(_LINKS[:n_links])) if n_links else ""
        proj_lines.append(f"{ptitle}{link_str} | Python, Docker | Mar 2024 -- Dec 2024")
        proj_lines.append("- Delivered automated CI evaluation harness raising branch test coverage from 62% to 91%.")

    # Education (1..2 stacked entries)
    edu_count = draw(st.integers(min_value=1, max_value=2))
    edu_lines = ["## Education"]
    schools = [
        ("ETH Z\u00fcrich, Z\u00fcrich, Switzerland", "M.Sc. in Distributed Systems, Jun 2022"),
        ("Imperial College London, London, UK", "B.Sc. in Computer Science, Jun 2020"),
    ]
    for sch, deg in schools[:edu_count]:
        edu_lines.extend([sch, deg])

    blocks = [exp_lines, proj_lines, edu_lines]
    perm = draw(st.permutations(range(len(blocks))))
    skills = [
        "## Technical Skills",
        "**Systems & Cloud**: Python,Go,Docker,Kubernetes,AWS,PostgreSQL,CI/CD,Terraform,OpenTelemetry",
    ]
    sections_md = ["\n".join(blocks[i]) for i in perm] + ["\n".join(skills)]
    return f"# {name}\n" + " | ".join(contact_parts) + "\n\n" + "\n\n".join(sections_md) + "\n"


_HYP_DB = DirectoryBasedExampleDatabase(Path(__file__).resolve().parents[1] / ".hypothesis" / "examples")


@pytest.mark.pdf_render
@seed(20261126)
@settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    database=_HYP_DB,
)
@given(md=_arbitrary_resume_md())
def test_hypothesis_arbitrary_resumes_pass_geometry_both_engines(md):
    """Property test: arbitrary long headers, 0-3 links, non-ASCII names, section orders,
    and stacked Education compile without margin/bottom/overlap violations in LaTeX & HTML."""
    if not HAVE_PDFLATEX:
        pytest.skip("pdflatex required for geometry property check")
    parsed = parse_markdown_resume(md)
    with tempfile.TemporaryDirectory() as d:
        tmp_dir = Path(d)
        res_l = render_compact_tiers(parsed, tmp_dir, "hyp_latex.pdf", target_pages=1)
        if res_l is None:
            tex = render_resume_tex(parsed, compact=True, compact_tier=3)
            l_pdf = latex_to_pdf(tex, tmp_dir, "hyp_latex.pdf")
        else:
            l_pdf = res_l["pdf_path"]
        assert l_pdf is not None
        lv = verify_pdf(l_pdf, expected_pages=pdf_page_count(l_pdf) or 1)
        geom_l = next(c for c in lv["checks"] if c["name"] == "geometry")
        assert geom_l["ok"], f"LaTeX geometry failure: {geom_l['detail']}"
        assert lv["pages"] <= 2

        if HAVE_BROWSER:
            res_h = render_density_tiers(parsed, tmp_dir, "hyp_html.pdf", theme="classic", target_pages=1)
            h_pdf = res_h["pdf_path"] if res_h else html_to_pdf(render_resume_html(parsed, theme="classic"), tmp_dir, "hyp_html.pdf")
            assert h_pdf is not None
            hv = verify_pdf(h_pdf, expected_pages=pdf_page_count(h_pdf) or 1)
            geom_h = next(c for c in hv["checks"] if c["name"] == "geometry")
            assert geom_h["ok"], f"HTML geometry failure: {geom_h['detail']}"
            assert hv["pages"] <= 2
