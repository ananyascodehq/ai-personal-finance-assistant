from pathlib import Path

import pytest

from backend.analysis.processor import (
    InvalidExpenseFile,
    check_columns,
    load_csv,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "test_data"


def load(name):
    return check_columns(load_csv(DATA / name))


def test_clean_file_loads():
    df = load("01_nov_clean.csv")
    assert list(df.columns) == ["Date", "Category", "Amount", "Description"]
    assert len(df) == 45


def test_doc_example_has_three_rows():
    assert len(load("03_doc_example.csv")) == 3


def test_shuffled_columns_are_reordered():
    df = load("13_shuffled_column_order.csv")
    assert list(df.columns) == ["Date", "Category", "Amount", "Description"]


def test_wrong_column_names_raise_error():
    with pytest.raises(InvalidExpenseFile):
        load("11_wrong_column_names.csv")


def test_missing_column_names_the_column():
    with pytest.raises(InvalidExpenseFile, match="Description"):
        load("12_missing_column.csv")


def test_header_only_file_gives_empty_table():
    assert load("08_empty_only_header.csv").empty