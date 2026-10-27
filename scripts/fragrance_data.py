"""Loading and cleaning of the Fragrantica `perfumes.csv` dump for ingestion."""

from datetime import date

import pandas as pd

MIN_VOTES = 10
MIN_YEAR = 1700

# The dump uses female/male; the frontend, rewrite prompt and SQL filter use women/men.
GENDER_MAP = {"female": "women", "male": "men", "unisex": "unisex"}


def parse_weighted(value) -> list[str]:
    """Turn 'Orange:100|Bergamot:28' into ['orange', 'bergamot'].

    Weights are discarded and low-weight items are kept on purpose: faint notes
    are often more present than the community weighting suggests.
    """
    if not isinstance(value, str) or not value.strip():
        return []
    items = []
    for part in value.split("|"):
        name, sep, weight = part.rpartition(":")
        if not sep or not weight.strip().replace(".", "", 1).isdigit():
            name = part
        name = name.strip().lower()
        if name and name not in items:
            items.append(name)
    return items


def clean_year(value) -> int | None:
    if pd.isna(value):
        return None
    year = int(value)
    return year if MIN_YEAR <= year <= date.today().year + 1 else None


def build_embedding_text(gender: str, notes: list[str], accords: list[str]) -> str:
    return f"Gender: {gender}. Notes: {', '.join(notes)}. Accords: {', '.join(accords)}"


def load_fragrances(csv_path: str, min_votes: int = MIN_VOTES) -> pd.DataFrame:
    """Read the dump and return one clean row per (name, brand), ready to embed.

    Keeps rows with at least `min_votes` votes, at least one accord and at least
    one note. Output columns: name, brand, gender, rating, rating_count, year,
    top_notes, middle_notes, base_notes, main_accords, url, embedding_text.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["vote_count"].fillna(0) >= min_votes]
    df = df.sort_values("vote_count", ascending=False)
    df = df.drop_duplicates(["name", "brand"], keep="first")

    layers = {
        col: df[f"notes_{col}"].map(parse_weighted)
        for col in ("top", "middle", "base", "flat")
    }
    accords = df["accords"].map(parse_weighted)
    gender = df["gender"].str.strip().str.lower().map(GENDER_MAP)

    # Rows with only an unlayered note list fall back to it as their top notes.
    has_layers = (
        layers["top"].str.len() + layers["middle"].str.len() + layers["base"].str.len()
    ) > 0
    layers["top"] = layers["top"].where(has_layers, layers["flat"])

    all_notes = layers["top"] + layers["middle"] + layers["base"]
    keep = (all_notes.str.len() > 0) & (accords.str.len() > 0) & gender.notna()
    df, accords, gender, all_notes = df[keep], accords[keep], gender[keep], all_notes[keep]

    out = pd.DataFrame(
        {
            "name": df["name"].str.strip(),
            "brand": df["brand"].str.strip(),
            "gender": gender,
            "rating": df["rating_avg"],
            "rating_count": df["vote_count"].astype(int),
            "year": df["year"].map(clean_year).astype(object),
            "top_notes": layers["top"][keep].map(", ".join),
            "middle_notes": layers["middle"][keep].map(", ".join),
            "base_notes": layers["base"][keep].map(", ".join),
            "main_accords": accords.map(", ".join),
            "url": df["url"],
        }
    )
    out["embedding_text"] = [
        build_embedding_text(g, n, a) for g, n, a in zip(gender, all_notes, accords)
    ]
    return out.reset_index(drop=True)
