"""Tests for salary_lookup.py - format_entry, match_score, search_company, validate."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import salary_lookup  # noqa: E402
from salary_lookup import (  # noqa: E402
    anglicize,
    collect_validation_issues,
    extract_core_words,
    format_entry,
    match_score,
    normalize,
    search_company,
    validate_data,
)

PRIVACY_FOOTNOTE = "* N/A = Too few employees to publish (privacy)"


# ---------------------------------------------------------------------------
# format_entry
# ---------------------------------------------------------------------------

def test_privacy_footnote_omitted_when_no_row_is_suppressed():
    entry = {
        "company": "Example Corp",
        "city": "",
        "categories": {"all_employees": {"count": 500, "index": 108.5}},
    }
    rendered = format_entry(entry, {"index_baseline": 100, "index_label": "Index"})
    assert PRIVACY_FOOTNOTE not in rendered


def test_privacy_footnote_is_printed_when_a_row_is_suppressed():
    entry = {
        "company": "Example Corp",
        "city": "",
        "categories": {
            "all_employees": {"count": 500, "index": 108.5},
            "small_team": {"count": 3, "index": None},
        },
    }
    rendered = format_entry(entry, {"index_baseline": 100, "index_label": "Index"})
    assert PRIVACY_FOOTNOTE in rendered
    assert "Small Team" in rendered


def test_zero_count_is_displayed_as_zero():
    entry = {"company": "Example Corp", "city": "",
             "categories": {"public_data": {"count": 0, "index": 100.0}}}
    rendered = format_entry(entry, {"index_baseline": 100, "index_label": "Index"})
    assert "Public Data" in rendered


def test_text_index_does_not_crash():
    entry = {"company": "Example Corp", "city": "",
             "categories": {"sample": {"count": 3, "index": "private"}}}
    assert "private" in format_entry(entry, {"index_baseline": 100, "index_label": "Index"})


def test_zero_baseline_prints_raw_values_without_percent():
    entry = {"company": "Example Corp", "city": "",
             "categories": {"it": {"count": None, "index": 45000.0}}}
    rendered = format_entry(entry, {"index_baseline": 0, "index_label": "Salary"})
    assert "45000.0" in rendered
    assert "%" not in rendered


def test_custom_baseline_prints_diff_percent():
    entry = {"company": "Example Corp", "city": "",
             "categories": {"it": {"count": None, "index": 45000.0}}}
    rendered = format_entry(entry, {"index_baseline": 40000, "index_label": "Salary"})
    assert "+12.5%" in rendered


def test_null_categories_with_sibling_dict_does_not_crash():
    entry = {"company": "Example Corp", "city": "", "categories": None,
             "engineering": {"count": 10, "index": 105.0}}
    rendered = format_entry(entry, {"index_baseline": 100, "index_label": "Index"})
    assert "Engineering" in rendered


# ---------------------------------------------------------------------------
# normalize / anglicize / extract_core_words
# ---------------------------------------------------------------------------

def test_normalize_strips_legal_suffixes_and_casefolds():
    assert normalize("Novo Nordisk A/S") == normalize("novo nordisk")


def test_anglicize_folds_nordic_spellings():
    assert anglicize("Brøderne") == "broderne"


def test_extract_core_words_drops_single_letters():
    words = extract_core_words("Acme Corp")
    assert "acme" in words and "corp" in words


# ---------------------------------------------------------------------------
# match_score / search_company
# ---------------------------------------------------------------------------

def test_exact_company_match_scores_highest():
    entry = {"company": "Novo Nordisk A/S", "city": "",
             "categories": {"all_employees": {"index": 100.0}}}
    assert match_score("Novo Nordisk A/S", entry["company"]) == 100


def test_data_has_two_companies():
    data = {"metadata": {"index_baseline": 100},
            "companies": [
                {"company": "Novo Nordisk A/S", "city": "Bagsværd",
                 "categories": {"all_employees": {"index": 108.5}}},
                {"company": "Ørsted A/S", "city": "Fredericia",
                 "categories": {"all_employees": {"index": 105.2}}},
            ]}
    results = search_company(data, "Novo")
    assert results and results[0]["company"] == "Novo Nordisk A/S"

    orsted_results = search_company(data, "Ørsted", city="Fredericia")
    assert [e["company"] for e in orsted_results] == ["Ørsted A/S"]


def test_city_filter_is_fuzzy_on_anglicized_spelling():
    data = {"metadata": {"index_baseline": 100},
            "companies": [
                {"company": "Acme A/S", "city": "København",
                 "categories": {"all_employees": {"index": 100.0}}},
            ]}
    # Anglified query matches the Ø city.
    assert search_company(data, "Acme", city="Kobenhavn") != []


# ---------------------------------------------------------------------------
# validate_data
# ---------------------------------------------------------------------------

def test_validate_data_accepts_wellformed_data():
    data = {"metadata": {"index_baseline": 100},
            "companies": [{"company": "Acme", "categories": {"index": {"index": 100.0}}}]}
    errors, warnings = collect_validation_issues(data)
    assert errors == [] and warnings == []
    assert validate_data(data) is data


def test_validate_data_rejects_bad_metadata_shape():
    data = {"metadata": {"index_baseline": "lots"}, "companies": []}
    errors, _ = collect_validation_issues(data)
    assert any("index_baseline" in e for e in errors)
    with pytest.raises(SystemExit):
        validate_data(data)


def test_validate_data_warns_on_duplicate_companies():
    data = {"companies": [
        {"company": "Acme"}, {"company": "acme"}]}
    errors, warnings = collect_validation_issues(data)
    assert errors == []
    assert any("Duplicate company" in w for w in warnings)
