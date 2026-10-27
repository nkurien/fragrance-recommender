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

# Captured at import, before the autouse fixture swaps record_usage for a stub
_real_record_usage = main_module.record_usage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def no_usage_writes(monkeypatch):
    """Never let tests write to the real usage_log table via the module-level pool."""
    stub = MagicMock()
    monkeypatch.setattr(main_module, "record_usage", stub)
    return stub


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


def _raw(parsed, headers: dict | None = None):
    """Wrap a parsed Groq response like the SDK's with_raw_response does."""
    raw = MagicMock()
    raw.parse.return_value = parsed
    raw.headers = headers or {}
    return raw


def _mock_groq_response(content: str, headers: dict | None = None):
    """Mock a non-streaming Groq response (used for the rewrite call)."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return _raw(response, headers)


def _mock_groq_stream(text: str, usage=None, headers: dict | None = None):
    """Mock a streaming Groq response (used for the recommendation call)."""
    delta = MagicMock()
    delta.content = text
    choice = MagicMock()
    choice.delta = delta
    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = usage
    return _raw(iter([chunk]), headers)


def _parse_stream(response) -> tuple[list, str]:
    """Parse an NDJSON streaming response into (matches, recommendation_text)."""
    matches = []
    tokens = []
    for line in response.text.strip().split("\n"):
        if not line.strip():
            continue
        event = json.loads(line)
        if event["type"] == "matches":
            matches = event["data"]
        elif event["type"] == "token":
            tokens.append(event["text"])
    return matches, "".join(tokens)


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
    mock_client.chat.completions.with_raw_response.create.return_value = _mock_groq_response(mock_json)

    with patch("main.groq_client", mock_client):
        result = rewrite_query(description, history=[])

    assert result == expected


def test_rewrite_query_exception_falls_back_to_raw():
    mock_client = MagicMock()
    mock_client.chat.completions.with_raw_response.create.side_effect = Exception("timeout")

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
    mock_client.chat.completions.with_raw_response.create.return_value = _mock_groq_response(
        '{"embedding_text": "Notes: rose", "brand_filter": null, "exclude_name": null}'
    )

    with patch("main.groq_client", mock_client):
        rewrite_query("test query", history=history_turns)

    assert mock_client.chat.completions.with_raw_response.create.call_count == expected_call_count
    call_messages = mock_client.chat.completions.with_raw_response.create.call_args[1]["messages"]
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

    # First Groq call → rewrite_query; second → streaming recommendation
    mock_groq.chat.completions.with_raw_response.create.side_effect = [
        _mock_groq_response(json.dumps(rewrite_result)),
        _mock_groq_stream("Here is my recommendation."),
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


def test_recommend_returns_500_when_db_pool_none(monkeypatch):
    """Safety-net: if db_pool is somehow None, the endpoint must return 500 not crash."""
    monkeypatch.setattr(main_module, "db_pool", None)
    monkeypatch.setattr(main_module, "groq_client", MagicMock())
    client = TestClient(main_module.app)
    response = client.post("/api/recommend", json={"description": "floral"})
    assert response.status_code == 500
    assert "uninitialized" in response.json()["detail"].lower()


def test_match_score_never_negative(client_with_mocks):
    """Distance > 1.0 (worst-case cosine) must not produce a negative match_score."""
    mock_groq, mock_cur, client = client_with_mocks

    mock_cur.fetchall.return_value = [make_row(name="bad-match", distance=1.5)]
    mock_groq.chat.completions.with_raw_response.create.side_effect = [
        _mock_groq_response(
            '{"embedding_text": "Notes: rose", "brand_filter": null, "exclude_name": null}'
        ),
        _mock_groq_stream("Here are my picks."),
    ]

    response = client.post("/api/recommend", json={"description": "floral"})
    assert response.status_code == 200
    matches, _ = _parse_stream(response)
    assert matches[0]["match_score"] >= 0


def test_llm_receives_at_most_7_candidates(client_with_mocks):
    """Ensure the recommendation prompt is capped at 7 candidates even when 30 rows are returned."""
    mock_groq, mock_cur, client = client_with_mocks

    # Return 30 distinct rows from the DB
    mock_cur.fetchall.return_value = [
        make_row(name=f"frag-{i}", brand="brand-x", distance=0.1 + i * 0.01)
        for i in range(30)
    ]

    mock_groq.chat.completions.with_raw_response.create.side_effect = [
        _mock_groq_response(
            '{"embedding_text": "Notes: rose", "brand_filter": null, "exclude_name": null}'
        ),
        _mock_groq_stream("Here are my picks."),
    ]

    response = client.post("/api/recommend", json={"description": "something floral"})
    assert response.status_code == 200

    # The second Groq call is the recommendation; grab its user-message content
    second_call_messages = mock_groq.chat.completions.with_raw_response.create.call_args_list[1][1]["messages"]
    user_message = next(m["content"] for m in second_call_messages if m["role"] == "user")

    # Count how many numbered candidate entries appear in the prompt
    import re
    candidate_entries = re.findall(r"^\d+\. \*\*", user_message, re.MULTILINE)
    assert len(candidate_entries) <= 7

    # Frontend still gets all 30
    matches, _ = _parse_stream(response)
    assert len(matches) == 30


def test_model_ids_and_reasoning_budget(client_with_mocks):
    """Both stages use the gpt-oss models with low reasoning effort and room for reasoning tokens."""
    mock_groq, _, client = client_with_mocks
    mock_groq.chat.completions.with_raw_response.create.side_effect = [
        _mock_groq_response(
            '{"embedding_text": "Notes: rose", "brand_filter": null, "exclude_name": null}'
        ),
        _mock_groq_stream("Here are my picks."),
    ]

    client.post("/api/recommend", json={"description": "something floral"})

    rewrite_kwargs, recommend_kwargs = (
        c[1] for c in mock_groq.chat.completions.with_raw_response.create.call_args_list
    )
    assert rewrite_kwargs["model"] == "openai/gpt-oss-20b"
    assert recommend_kwargs["model"] == "openai/gpt-oss-120b"
    for kwargs in (rewrite_kwargs, recommend_kwargs):
        assert kwargs["reasoning_effort"] == "low"
    assert rewrite_kwargs["max_tokens"] >= 600
    assert recommend_kwargs["max_tokens"] >= 1500


def test_non_breaking_hyphens_normalised_in_stream(client_with_mocks):
    mock_groq, _, client = client_with_mocks
    mock_groq.chat.completions.with_raw_response.create.side_effect = [
        _mock_groq_response(
            '{"embedding_text": "Notes: rose", "brand_filter": null, "exclude_name": null}'
        ),
        _mock_groq_stream("**bal\u2011d\u2011afrique** is lovely."),
    ]

    response = client.post("/api/recommend", json={"description": "something floral"})
    _, text = _parse_stream(response)
    assert text == "**bal-d-afrique** is lovely."


def test_empty_completion_falls_back_to_top_matches(client_with_mocks):
    """If reasoning eats the token budget and no prose streams, the user still gets the matches."""
    mock_groq, _, client = client_with_mocks
    mock_groq.chat.completions.with_raw_response.create.side_effect = [
        _mock_groq_response(
            '{"embedding_text": "Notes: rose", "brand_filter": null, "exclude_name": null}'
        ),
        _mock_groq_stream(""),
    ]

    response = client.post("/api/recommend", json={"description": "something floral"})
    _, text = _parse_stream(response)
    assert "**bal-d-afrique** by byredo" in text


# ---------------------------------------------------------------------------
# Health check connection leak
# ---------------------------------------------------------------------------


def test_health_check_putconn_called_on_db_error(monkeypatch):
    """putconn must be called even when the DB query raises, so the connection
    is returned to the pool and doesn't leak."""
    mock_conn = MagicMock()
    mock_conn.cursor.side_effect = Exception("db is down")

    mock_pool = MagicMock()
    mock_pool.getconn.return_value = mock_conn

    monkeypatch.setattr(main_module, "db_pool", mock_pool)
    monkeypatch.setattr(main_module, "groq_client", MagicMock())

    client = TestClient(main_module.app)
    response = client.get("/api/health")

    assert response.status_code == 200
    assert "error" in response.json()["database"]
    mock_pool.putconn.assert_called_once_with(mock_conn)


# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------

RATE_HEADERS = {
    "x-ratelimit-limit-requests": "1000",
    "x-ratelimit-remaining-requests": "990",
    "x-ratelimit-limit-tokens": "8000",
    "x-ratelimit-remaining-tokens": "7500",
}


def _usage(prompt, completion):
    u = MagicMock()
    u.prompt_tokens = prompt
    u.completion_tokens = completion
    u.total_tokens = prompt + completion
    return u


def test_record_usage_row_contents(monkeypatch):
    executor = MagicMock()
    monkeypatch.setattr(main_module, "_usage_executor", executor)

    _real_record_usage("rewrite", "openai/gpt-oss-20b", _usage(385, 144), 877, RATE_HEADERS)

    fn, row = executor.submit.call_args[0]
    assert fn is main_module._insert_usage
    assert row == ("rewrite", "openai/gpt-oss-20b", 385, 144, 877, 1000, 990, 8000, 7500)


def test_record_usage_tolerates_missing_usage_and_headers(monkeypatch):
    executor = MagicMock()
    monkeypatch.setattr(main_module, "_usage_executor", executor)

    _real_record_usage("recommend", "openai/gpt-oss-120b", None, 1200, {})

    _, row = executor.submit.call_args[0]
    assert row == ("recommend", "openai/gpt-oss-120b", None, None, 1200, None, None, None, None)


def test_record_usage_never_raises(monkeypatch):
    executor = MagicMock()
    executor.submit.side_effect = RuntimeError("executor shut down")
    monkeypatch.setattr(main_module, "_usage_executor", executor)

    _real_record_usage("rewrite", "m", _usage(1, 1), 5, {})  # must not raise


def test_rewrite_records_usage(no_usage_writes):
    mock_client = MagicMock()
    mock_client.chat.completions.with_raw_response.create.return_value = _mock_groq_response(
        '{"embedding_text": "Notes: rose", "brand_filter": null, "exclude_name": null}',
        headers=RATE_HEADERS,
    )
    parsed = mock_client.chat.completions.with_raw_response.create.return_value.parse.return_value
    parsed.usage = _usage(385, 144)

    with patch("main.groq_client", mock_client):
        rewrite_query("rose", history=[])

    stage, model, usage, latency, headers = no_usage_writes.call_args[0]
    assert (stage, model) == ("rewrite", "openai/gpt-oss-20b")
    assert (usage.prompt_tokens, usage.completion_tokens) == (385, 144)
    assert headers == RATE_HEADERS
    assert isinstance(latency, int)


def test_recommend_records_streamed_usage(client_with_mocks, no_usage_writes):
    mock_groq, _, client = client_with_mocks
    mock_groq.chat.completions.with_raw_response.create.side_effect = [
        _mock_groq_response(
            '{"embedding_text": "Notes: rose", "brand_filter": null, "exclude_name": null}'
        ),
        _mock_groq_stream("Here are my picks.", usage=_usage(900, 300), headers=RATE_HEADERS),
    ]

    client.post("/api/recommend", json={"description": "something floral"})

    recommend_call = next(
        c for c in no_usage_writes.call_args_list if c[0][0] == "recommend"
    )
    _, model, usage, _, headers = recommend_call[0]
    assert model == "openai/gpt-oss-120b"
    assert (usage.prompt_tokens, usage.completion_tokens) == (900, 300)
    assert headers == RATE_HEADERS


def _pool_with_conn():
    conn = MagicMock()
    pool = MagicMock()
    pool.getconn.return_value = conn
    return pool, conn


def test_insert_usage_commits_and_returns_connection(monkeypatch):
    pool, conn = _pool_with_conn()
    monkeypatch.setattr(main_module, "db_pool", pool)
    row = ("rewrite", "m", 1, 2, 3, None, None, None, None)

    main_module._insert_usage(row)

    conn.cursor.return_value.__enter__.return_value.execute.assert_called_once()
    conn.commit.assert_called_once()
    pool.putconn.assert_called_once_with(conn)


def test_insert_usage_failure_rolls_back_and_returns_connection(monkeypatch):
    pool, conn = _pool_with_conn()
    conn.cursor.return_value.__enter__.return_value.execute.side_effect = Exception("no table")
    monkeypatch.setattr(main_module, "db_pool", pool)

    main_module._insert_usage(("rewrite", "m", 1, 2, 3, None, None, None, None))  # no raise

    conn.rollback.assert_called_once()
    pool.putconn.assert_called_once_with(conn)


def test_ensure_usage_table_runs_ddl_and_returns_connection(monkeypatch):
    pool, conn = _pool_with_conn()
    monkeypatch.setattr(main_module, "db_pool", pool)

    main_module.ensure_usage_table()

    sql = conn.cursor.return_value.__enter__.return_value.execute.call_args[0][0]
    assert "CREATE TABLE IF NOT EXISTS usage_log" in sql
    conn.commit.assert_called_once()
    pool.putconn.assert_called_once_with(conn)


def test_ensure_usage_table_swallows_errors(monkeypatch):
    pool, conn = _pool_with_conn()
    conn.cursor.return_value.__enter__.return_value.execute.side_effect = Exception("denied")
    monkeypatch.setattr(main_module, "db_pool", pool)

    main_module.ensure_usage_table()  # app must still start

    pool.putconn.assert_called_once_with(conn)


def test_usage_summary_shapes_rows(monkeypatch):
    from datetime import date, datetime, timezone

    monkeypatch.setattr(main_module.settings, "daily_token_limit", 200_000)
    ts = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)
    cur = MagicMock()
    cur.fetchall.side_effect = [
        # 24h aggregates: model, count, prompt, completion, avg latency
        [("openai/gpt-oss-120b", 4, 3600, 1200, 1100.4), ("openai/gpt-oss-20b", 4, 1500, 500, 600.0)],
        # latest limits: model, ts, limit_req, remaining_req, limit_tok, remaining_tok
        [("openai/gpt-oss-120b", ts, 1000, 990, 8000, 7500)],
        # last 7 days
        [(date(2026, 10, 27), 8, 6800)],
    ]

    out = main_module._usage_summary(cur)

    assert out["daily_token_limit"] == 200_000
    big, small = out["models"]  # sorted by model name: 120b before 20b
    assert big["model"] == "openai/gpt-oss-120b"
    assert big["tokens_24h"] == 4800 and big["requests_24h"] == 4
    assert big["avg_latency_ms"] == 1100
    assert big["remaining_requests"] == 990 and big["limit_requests"] == 1000
    assert big["limits_as_of"] == ts.isoformat()
    assert small["remaining_requests"] is None  # no header snapshot yet
    assert out["days"] == [{"day": "2026-10-27", "requests": 8, "tokens": 6800}]


def test_usage_summary_includes_model_with_limits_but_no_recent_traffic():
    from datetime import datetime, timezone

    cur = MagicMock()
    cur.fetchall.side_effect = [
        [],
        [("openai/gpt-oss-20b", datetime(2026, 10, 26, tzinfo=timezone.utc), 1000, 950, 8000, 8000)],
        [],
    ]

    out = main_module._usage_summary(cur)

    (m,) = out["models"]
    assert m["requests_24h"] == 0 and m["tokens_24h"] == 0
    assert m["remaining_requests"] == 950


@pytest.fixture()
def usage_client(monkeypatch):
    monkeypatch.setattr(main_module.settings, "admin_key", "s3cret-key")
    pool, conn = _pool_with_conn()
    monkeypatch.setattr(main_module, "db_pool", pool)
    monkeypatch.setattr(
        main_module, "_usage_summary", lambda cur: {"daily_token_limit": 1, "models": [], "days": []}
    )
    return TestClient(main_module.app), pool, conn


def test_usage_endpoint_404_when_admin_key_unset(monkeypatch):
    monkeypatch.setattr(main_module.settings, "admin_key", "")
    client = TestClient(main_module.app)
    assert client.get("/api/usage", headers={"X-Admin-Key": "anything"}).status_code == 404


@pytest.mark.parametrize("headers", [{}, {"X-Admin-Key": "wrong"}, {"X-Admin-Key": ""}])
def test_usage_endpoint_401_without_valid_key(usage_client, headers):
    client, pool, _ = usage_client
    assert client.get("/api/usage", headers=headers).status_code == 401
    pool.getconn.assert_not_called()  # never touches the DB when unauthorised


def test_usage_endpoint_returns_summary_with_valid_key(usage_client):
    client, pool, conn = usage_client
    response = client.get("/api/usage", headers={"X-Admin-Key": "s3cret-key"})
    assert response.status_code == 200
    assert response.json() == {"daily_token_limit": 1, "models": [], "days": []}
    pool.putconn.assert_called_once_with(conn)


def test_usage_endpoint_500_and_returns_connection_on_db_error(usage_client, monkeypatch):
    client, pool, conn = usage_client

    def boom(cur):
        raise Exception("relation usage_log does not exist")

    monkeypatch.setattr(main_module, "_usage_summary", boom)
    response = client.get("/api/usage", headers={"X-Admin-Key": "s3cret-key"})
    assert response.status_code == 500
    assert "usage_log" not in response.text  # internals aren't leaked
    pool.putconn.assert_called_once_with(conn)


def test_usage_endpoint_handles_non_ascii_key(usage_client):
    client, _, _ = usage_client
    response = client.get("/api/usage", headers={"X-Admin-Key": "caf\u00e9".encode("latin-1")})
    assert response.status_code == 401
