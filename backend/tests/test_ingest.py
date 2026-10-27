"""
Unit tests for the perfumes.csv loader used by scripts/embed_and_upload.py.
Run with: pytest backend/tests/ -v
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts"))

from fragrance_data import (  # noqa: E402
    build_embedding_text,
    clean_year,
    load_fragrances,
    parse_weighted,
)


def make_csv(tmp_path, rows):
    base = {
        "name": "Test",
        "brand": "Brand",
        "year": 2020,
        "gender": "female",
        "url": "https://example.com",
        "rating_avg": 4.0,
        "vote_count": 50,
        "accords": "citrus:100|woody:40",
        "notes_top": "Orange:100|Bergamot:28",
        "notes_middle": "Jasmine:30",
        "notes_base": "Musk:10",
        "notes_flat": None,
    }
    df = pd.DataFrame([{**base, **r} for r in rows])
    path = tmp_path / "perfumes.csv"
    df.to_csv(path, index=False)
    return str(path)


class TestParseWeighted:
    def test_strips_weights_and_lowercases(self):
        assert parse_weighted("Orange:100|Bergamot:28") == ["orange", "bergamot"]

    def test_keeps_zero_weight_items(self):
        assert parse_weighted("Jasmine:30|Flowers:0") == ["jasmine", "flowers"]

    def test_item_without_weight(self):
        assert parse_weighted("Orange|Bergamot:28") == ["orange", "bergamot"]

    def test_colon_inside_name(self):
        assert parse_weighted("Note: Rose:12") == ["note: rose"]

    def test_dedupes_preserving_order(self):
        assert parse_weighted("Rose:10|Musk:5|rose:3") == ["rose", "musk"]

    @pytest.mark.parametrize("value", [None, float("nan"), "", "  "])
    def test_empty_values(self, value):
        assert parse_weighted(value) == []


class TestCleanYear:
    def test_valid(self):
        assert clean_year(2019.0) == 2019

    @pytest.mark.parametrize("value", [float("nan"), 20, 1500, 2099])
    def test_invalid_becomes_none(self, value):
        assert clean_year(value) is None


def test_embedding_text_format():
    text = build_embedding_text("women", ["rose", "musk"], ["floral", "powdery"])
    assert text == "Gender: women. Notes: rose, musk. Accords: floral, powdery"


class TestLoadFragrances:
    def test_basic_row(self, tmp_path):
        df = load_fragrances(make_csv(tmp_path, [{}]))
        row = df.iloc[0]
        assert row["gender"] == "women"
        assert row["top_notes"] == "orange, bergamot"
        assert row["main_accords"] == "citrus, woody"
        assert row["rating_count"] == 50
        assert row["embedding_text"] == (
            "Gender: women. Notes: orange, bergamot, jasmine, musk. Accords: citrus, woody"
        )

    def test_gender_mapping(self, tmp_path):
        rows = [
            {"name": "A", "gender": "female"},
            {"name": "B", "gender": "male"},
            {"name": "C", "gender": "unisex"},
        ]
        df = load_fragrances(make_csv(tmp_path, rows))
        assert dict(zip(df["name"], df["gender"])) == {
            "A": "women",
            "B": "men",
            "C": "unisex",
        }

    def test_vote_threshold_is_inclusive(self, tmp_path):
        rows = [
            {"name": "A", "vote_count": 9},
            {"name": "B", "vote_count": 10},
        ]
        df = load_fragrances(make_csv(tmp_path, rows), min_votes=10)
        assert list(df["name"]) == ["B"]

    def test_drops_rows_without_accords_or_notes(self, tmp_path):
        rows = [
            {"name": "NoAccords", "accords": None},
            {
                "name": "NoNotes",
                "notes_top": None,
                "notes_middle": None,
                "notes_base": None,
            },
            {"name": "Good"},
        ]
        df = load_fragrances(make_csv(tmp_path, rows))
        assert list(df["name"]) == ["Good"]

    def test_flat_notes_fallback_when_no_layers(self, tmp_path):
        rows = [
            {
                "notes_top": None,
                "notes_middle": None,
                "notes_base": None,
                "notes_flat": "Danish pastry:100|Raspberry:0",
            }
        ]
        row = load_fragrances(make_csv(tmp_path, rows)).iloc[0]
        assert row["top_notes"] == "danish pastry, raspberry"
        assert "Notes: danish pastry, raspberry." in row["embedding_text"]

    def test_flat_notes_ignored_when_layers_exist(self, tmp_path):
        rows = [{"notes_flat": "Something:100"}]
        row = load_fragrances(make_csv(tmp_path, rows)).iloc[0]
        assert "something" not in row["embedding_text"]

    def test_dedupes_name_brand_keeping_most_votes(self, tmp_path):
        rows = [
            {"vote_count": 20, "rating_avg": 3.0},
            {"vote_count": 80, "rating_avg": 4.5},
        ]
        df = load_fragrances(make_csv(tmp_path, rows))
        assert len(df) == 1
        assert df.iloc[0]["rating_count"] == 80

    def test_invalid_year_becomes_none(self, tmp_path):
        df = load_fragrances(make_csv(tmp_path, [{"year": 20}]))
        assert df.iloc[0]["year"] is None

    def test_embedding_text_excludes_name_and_brand(self, tmp_path):
        rows = [{"name": "UniqueName", "brand": "UniqueBrand"}]
        text = load_fragrances(make_csv(tmp_path, rows)).iloc[0]["embedding_text"]
        assert "UniqueName" not in text and "UniqueBrand" not in text
