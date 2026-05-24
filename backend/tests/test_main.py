"""
Unit tests for fragrance recommender backend.
Run with: pytest backend/tests/ -v
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

import main as main_module
from main import _build_conditions, _score_and_rank, rewrite_query

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_row(
    name="test-frag",
    brand="test-brand",
    gender="unisex",
    rating=4.0,
    rating_count=1000,
    year=2020,
    top_notes="bergamot",
    middle_notes="rose",
    base_notes="musk",
    main_accords="floral",
    url="https://example.com",
    distance=0.2,
):
    return (
        name,
        brand,
        gender,
        rating,
        rating_count,
        year,
        top_notes,
        middle_notes,
        base_notes,
        main_accords,
        url,
        distance,
    )


# ---------------------------------------------------------------------------
# _build_conditions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gender,brand_filter,exclude_name,expected_conditions,expected_params",
    [
        # No filters → empty
        (None, None, None, [], []),
        # Gender only
        ("women", None, None, ["LOWER(gender) = LOWER(%s)"], ["women"]),
        # Brand only (slug passthrough)
        (None, "byredo", None, ["LOWER(brand) LIKE LOWER(%s)"], ["%byredo%"]),
        # Brand with space → hyphenated in LIKE param
        (
            None,
            "carolina herrera",
            None,
            ["LOWER(brand) LIKE LOWER(%s)"],
            ["%carolina-herrera%"],
        ),
        # Exclude name only (slug passthrough)
        (
            None,
            None,
            "slow-dance",
            ["LOWER(name) NOT LIKE LOWER(%s)"],
            ["%slow-dance%"],
        ),
        # Exclude name with space → hyphenated
        (None, None, "good girl", ["LOWER(name) NOT LIKE LOWER(%s)"], ["%good-girl%"]),
        # Gender + brand
        (
            "men",
            "creed",
            None,
            ["LOWER(gender) = LOWER(%s)", "LOWER(brand) LIKE LOWER(%s)"],
            ["men", "%creed%"],
        ),
        # Gender + exclude
        (
            "women",
            None,
            "coco-mademoiselle",
            ["LOWER(gender) = LOWER(%s)", "LOWER(name) NOT LIKE LOWER(%s)"],
            ["women", "%coco-mademoiselle%"],
        ),
        # All three
        (
            "unisex",
            "le-labo",
            "santal-33",
            [
                "LOWER(gender) = LOWER(%s)",
                "LOWER(brand) LIKE LOWER(%s)",
                "LOWER(name) NOT LIKE LOWER(%s)",
            ],
            ["unisex", "%le-labo%", "%santal-33%"],
        ),
    ],
)
def test_build_conditions(
    gender, brand_filter, exclude_name, expected_conditions, expected_params
):
    conditions, params = _build_conditions(gender, brand_filter, exclude_name)
    assert conditions == expected_conditions
    assert params == expected_params


# ---------------------------------------------------------------------------
# _score_and_rank
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rows,expected_first_name",
    [
        # Lower distance (higher similarity) wins when rating/count are equal
        (
            [
                make_row(name="close", distance=0.1, rating=4.0, rating_count=1000),
                make_row(name="far", distance=0.5, rating=4.0, rating_count=1000),
            ],
            "close",
        ),
        # Higher rating_count wins when distance and rating are equal
        (
            [
                make_row(name="popular", distance=0.3, rating=4.0, rating_count=50000),
                make_row(name="obscure", distance=0.3, rating=4.0, rating_count=10),
            ],
            "popular",
        ),
        # Higher rating wins when distance and count are equal
        (
            [
                make_row(name="top-rated", distance=0.3, rating=5.0, rating_count=500),
                make_row(name="low-rated", distance=0.3, rating=2.0, rating_count=500),
            ],
            "top-rated",
        ),
        # None rating treated as 0, does not crash
        (
            [
                make_row(name="no-rating", distance=0.1, rating=None, rating_count=100),
                make_row(name="has-rating", distance=0.4, rating=4.5, rating_count=100),
            ],
            "no-rating",  # similarity advantage outweighs missing rating
        ),
        # None rating_count treated as 0, does not crash; large similarity gap overrides popularity
        (
            [
                make_row(name="no-count", distance=0.01, rating=4.0, rating_count=None),
                make_row(name="has-count", distance=0.9, rating=4.0, rating_count=5000),
            ],
            "no-count",
        ),
        # Single row — no division-by-zero
        (
            [make_row(name="only-one", distance=0.3, rating=3.5, rating_count=200)],
            "only-one",
        ),
    ],
)
def test_score_and_rank_first_place(rows, expected_first_name):
    scored = _score_and_rank(rows)
    assert scored[0][1][0] == expected_first_name


def test_score_and_rank_empty():
    assert _score_and_rank([]) == []


def test_score_and_rank_descending_order():
    rows = [
        make_row(name="a", distance=0.5, rating=3.0, rating_count=100),
        make_row(name="b", distance=0.1, rating=5.0, rating_count=50000),
        make_row(name="c", distance=0.3, rating=4.0, rating_count=1000),
    ]
    scored = _score_and_rank(rows)
    scores = [s for s, _ in scored]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize(
    "distance,rating,rating_count",
    [
        (0.0, 5.0, 100000),
        (1.0, 0.0, 0),
        (0.5, None, None),
        (0.99, 4.9, 1),
    ],
)
def test_score_and_rank_score_bounds(distance, rating, rating_count):
    rows = [make_row(distance=distance, rating=rating, rating_count=rating_count)]
    scored = _score_and_rank(rows)
    score = scored[0][0]
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# rewrite_query
# ---------------------------------------------------------------------------


def _mock_groq_response(content: str):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.mark.parametrize(
    "mock_json,description,expected",
    [
        # Full valid response
        (
            '{"embedding_text": "Notes: rose, oud", "brand_filter": "creed", "exclude_name": "aventus"}',
            "something like Aventus by Creed",
            {
                "embedding_text": "Notes: rose, oud",
                "brand_filter": "creed",
                "exclude_name": "aventus",
            },
        ),
        # Null brand and exclude
        (
            '{"embedding_text": "Notes: vanilla, musk", "brand_filter": null, "exclude_name": null}',
            "something warm and cozy",
            {
                "embedding_text": "Notes: vanilla, musk",
                "brand_filter": None,
                "exclude_name": None,
            },
        ),
        # Missing optional fields default to None
        (
            '{"embedding_text": "Notes: bergamot"}',
            "fresh citrus",
            {
                "embedding_text": "Notes: bergamot",
                "brand_filter": None,
                "exclude_name": None,
            },
        ),
        # Null/empty embedding_text falls back to raw description
        (
            '{"embedding_text": null, "brand_filter": null, "exclude_name": null}',
            "my fallback description",
            {
                "embedding_text": "my fallback description",
                "brand_filter": None,
                "exclude_name": None,
            },
        ),
    ],
)
def test_rewrite_query_parsing(mock_json, description, expected):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_groq_response(mock_json)

    with patch("main.groq_client", mock_client):
        result = rewrite_query(description, history=[])

    assert result == expected


def test_rewrite_query_exception_falls_back_to_raw():
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("timeout")

    with patch("main.groq_client", mock_client):
        result = rewrite_query("woody and dark", history=[])

    assert result == {
        "embedding_text": "woody and dark",
        "brand_filter": None,
        "exclude_name": None,
    }


@pytest.mark.parametrize(
    "history_turns,expected_call_count",
    [
        # Empty history
        ([], 1),
        # 4 turns (within 6-turn window)
        (
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "more"},
                {"role": "assistant", "content": "sure"},
            ],
            1,
        ),
    ],
)
def test_rewrite_query_passes_history(history_turns, expected_call_count):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_groq_response(
        '{"embedding_text": "Notes: rose", "brand_filter": null, "exclude_name": null}'
    )

    with patch("main.groq_client", mock_client):
        rewrite_query("test query", history=history_turns)

    assert mock_client.chat.completions.create.call_count == expected_call_count
    call_messages = mock_client.chat.completions.create.call_args[1]["messages"]
    # system + history + user query
    assert len(call_messages) == 1 + len(history_turns) + 1


# ---------------------------------------------------------------------------
# brand_filter nullification when exclude_name is set  (the slow-dance fix)
# ---------------------------------------------------------------------------


@pytest.fixture()
def client_with_mocks(monkeypatch):
    """TestClient with DB pool and Groq fully mocked out."""
    # Mock embedding model — encode().tolist() must return a plain list
    mock_embed = MagicMock()
    mock_embed.encode.return_value.tolist.return_value = [0.0] * 384
    monkeypatch.setattr(main_module, "embedding_model", mock_embed)

    # Mock Groq client (used by both rewrite_query and the recommendation step)
    mock_groq = MagicMock()
    monkeypatch.setattr(main_module, "groq_client", mock_groq)

    # Mock DB pool returning a fixed set of candidate rows
    fake_row = make_row(name="bal-d-afrique", brand="byredo", distance=0.15)
    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = [fake_row]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    mock_pool = MagicMock()
    mock_pool.getconn.return_value = mock_conn
    monkeypatch.setattr(main_module, "db_pool", mock_pool)

    return mock_groq, mock_cur, TestClient(main_module.app)


@pytest.mark.parametrize(
    "rewrite_result,expect_brand_in_sql",
    [
        # "similar to" query: LLM returns brand + exclude → brand must NOT filter SQL
        (
            {
                "embedding_text": "Notes: rose",
                "brand_filter": "byredo",
                "exclude_name": "slow-dance",
            },
            False,
        ),
        # Explicit brand request (no exclude): brand SHOULD filter SQL
        (
            {
                "embedding_text": "Notes: rose",
                "brand_filter": "byredo",
                "exclude_name": None,
            },
            True,
        ),
        # No brand, no exclude
        (
            {
                "embedding_text": "Notes: rose",
                "brand_filter": None,
                "exclude_name": None,
            },
            False,
        ),
    ],
)
def test_brand_filter_nullified_when_exclude_name_set(
    client_with_mocks, rewrite_result, expect_brand_in_sql
):
    mock_groq, mock_cur, client = client_with_mocks

    # First Groq call → rewrite_query; second → recommendation text
    mock_groq.chat.completions.create.side_effect = [
        _mock_groq_response(json.dumps(rewrite_result)),
        _mock_groq_response("Here is my recommendation."),
    ]

    client.post(
        "/api/recommend",
        json={"description": "show me fragrances like slow dance by byredo"},
    )

    # mock_cur is the cursor used during the request — check the SQL it received
    executed_query = mock_cur.execute.call_args[0][0]

    # The SELECT always contains "brand" as a column; check for the WHERE filter condition
    if expect_brand_in_sql:
        assert "lower(brand) like" in executed_query.lower()
    else:
        assert "lower(brand) like" not in executed_query.lower()


# ---------------------------------------------------------------------------
# LLM candidate cap
# ---------------------------------------------------------------------------


def test_match_score_never_negative(client_with_mocks):
    """Distance > 1.0 (worst-case cosine) must not produce a negative match_score."""
    mock_groq, mock_cur, client = client_with_mocks

    mock_cur.fetchall.return_value = [make_row(name="bad-match", distance=1.5)]
    mock_groq.chat.completions.create.side_effect = [
        _mock_groq_response(
            '{"embedding_text": "Notes: rose", "brand_filter": null, "exclude_name": null}'
        ),
        _mock_groq_response("Here are my picks."),
    ]

    response = client.post("/api/recommend", json={"description": "floral"})
    assert response.status_code == 200
    match_score = response.json()["matches"][0]["match_score"]
    assert match_score >= 0


def test_llm_receives_at_most_7_candidates(client_with_mocks):
    """Ensure the recommendation prompt is capped at 7 candidates even when 30 rows are returned."""
    mock_groq, mock_cur, client = client_with_mocks

    # Return 30 distinct rows from the DB
    mock_cur.fetchall.return_value = [
        make_row(name=f"frag-{i}", brand="brand-x", distance=0.1 + i * 0.01)
        for i in range(30)
    ]

    mock_groq.chat.completions.create.side_effect = [
        _mock_groq_response(
            '{"embedding_text": "Notes: rose", "brand_filter": null, "exclude_name": null}'
        ),
        _mock_groq_response("Here are my picks."),
    ]

    response = client.post("/api/recommend", json={"description": "something floral"})
    assert response.status_code == 200

    # The second Groq call is the recommendation; grab its user-message content
    second_call_messages = mock_groq.chat.completions.create.call_args_list[1][1]["messages"]
    user_message = next(m["content"] for m in second_call_messages if m["role"] == "user")

    # Count how many numbered candidate entries appear in the prompt
    import re
    candidate_entries = re.findall(r"^\d+\. \*\*", user_message, re.MULTILINE)
    assert len(candidate_entries) <= 7

    # Frontend still gets all 30
    assert len(response.json()["matches"]) == 30
