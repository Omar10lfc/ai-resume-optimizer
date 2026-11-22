"""Tests for tools/convert_salary_excel.py - parsers, parse_sheet, CLI roundtrip.

Parser cases are adapted from the MIT ai-job-search repo's tests/test_convert_salary_excel.py.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import openpyxl  # noqa: E402
import tools.convert_salary_excel as salary_excel  # noqa: E402
from tools.convert_salary_excel import (  # noqa: E402
    COUNT_PATTERNS,
    INDEX_PATTERNS,
    parse_numeric_cell,
    parse_sheet,
    split_combined_company_city,
    header_matches,
)

TOOL = Path(__file__).resolve().parents[1] / "tools" / "convert_salary_excel.py"


def _workbook(rows):
    wb = openpyxl.Workbook()
    ws = wb.active
    for row in rows:
        ws.append(list(row))
    return ws


# ---------------------------------------------------------------------------
# parse_numeric_cell
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    (42, 42.0),
    ("42", 42.0),
    (1.5, 1.5),
    ("1,234.56", 1234.56),
    ("1.234,56", 1234.56),
    ("1,234,567.89", 1234567.89),
    ("1.234.567,89", 1234567.89),
])
def test_parse_numeric_cell(raw, expected):
    assert parse_numeric_cell(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["1,234", "1.234", True, "abc", None])
def test_parse_numeric_cell_rejects(raw):
    with pytest.raises((ValueError, TypeError)):
        parse_numeric_cell(raw)


# ---------------------------------------------------------------------------
# split_combined_company_city
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cell,expected", [
    ("Ørsted A/S,Fredericia", ("Ørsted A/S", "Fredericia")),
    ("Rambøll Danmark A/S, Odense,Odense C", ("Rambøll Danmark A/S, Odense", "Odense C")),
    ("Acme Corp", ("Acme Corp", "")),
    ("", ("", "")),
])
def test_split_combined_company_city(cell, expected):
    assert split_combined_company_city(cell) == expected


# ---------------------------------------------------------------------------
# header_matches
# ---------------------------------------------------------------------------

def test_header_matches_salary_index_token():
    assert header_matches("Salary Index", INDEX_PATTERNS)


def test_header_matches_salary_word_alone():
    assert header_matches("Median salary", INDEX_PATTERNS)


def test_header_matches_compound_glued_word():
    assert header_matches("lønindeks", INDEX_PATTERNS)


def test_header_matches_glued_non_compound_word_is_not_a_match():
    assert not header_matches("salaryindex", INDEX_PATTERNS)


# ---------------------------------------------------------------------------
# parse_sheet
# ---------------------------------------------------------------------------

def test_parse_sheet_pairs_count_and_index_columns():
    ws = _workbook([["Company", "Count", "Salary"], ["Acme", 120, 105.5]])
    companies = parse_sheet(ws)
    assert [c["company"] for c in companies] == ["Acme"]
    assert companies[0]["categories"]["all_employees"] == {
        "count": 120, "index": 105.5}


def test_parse_sheet_skips_id_column():
    ws = _workbook([["Company", "Id", "Salary Index"], ["Acme", 99, 108.5]])
    companies = parse_sheet(ws)
    assert [c["company"] for c in companies] == ["Acme"]
    assert "salary_index" in companies[0]["categories"]


def test_parse_sheet_keeps_notes_column_out_of_salary_categories():
    ws = _workbook([["Company", "Salary Index", "Notes"],
                    ["Acme", 105.5, "quarterly average"]])
    companies = parse_sheet(ws)
    assert [c["company"] for c in companies] == ["Acme"]
    assert "notes" not in companies[0]["categories"]


def test_parse_sheet_handles_wrapped_header_rows():
    ws = _workbook([["Company", "City", "Employees", "Median\nsalary"],
                    ["Acme", "Copenhagen", 12, 105.5]])
    companies = parse_sheet(ws)
    assert [c["company"] for c in companies] == ["Acme"]


# ---------------------------------------------------------------------------
# CLI roundtrip
# ---------------------------------------------------------------------------

def test_convert_cli_writes_salary_data_json(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Company", "Count", "Salary"])
    ws.append(["Acme", 120, 105.5])
    xlsx = tmp_path / "salary.xlsx"
    wb.save(xlsx)
    out = tmp_path / "salary_data.json"

    proc = subprocess.run(
        [sys.executable, str(TOOL), str(xlsx), "--output", str(out)],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["metadata"]["source"] == "salary"
    assert data["companies"][0]["categories"]["all_employees"] == {
        "count": 120, "index": 105.5}
