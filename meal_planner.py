import streamlit as st
import pandas as pd
import numpy as np
import re
import textwrap

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
from itertools import combinations


# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Meal Planner",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Google Sheet CSV URLs ───────────────────────────────────────────────────
# Main recipe sheet
GOOGLE_SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vQPVOLW-IYb4T3HojVWtCxgXw1wmXl4TQSU1QAuDRc9A-o0h36DOXS5Rp6YagT-E2YB6Z0P1tcaxOj5/"
    "pub?gid=0&single=true&output=csv"
)

# Price sheet
# Replace this with the published CSV link for your Price tab.
PRICE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQPVOLW-IYb4T3HojVWtCxgXw1wmXl4TQSU1QAuDRc9A-o0h36DOXS5Rp6YagT-E2YB6Z0P1tcaxOj5/pub?gid=1515688392&single=true&output=csv"


# ── Colour scheme ───────────────────────────────────────────────────────────
COLOURS = {
    "Tried": "#4CAF93",
    "Not Tried": "#E8885A",
    "bg": "#FAF7F2",
    "card": "#FFFFFF",
    "accent": "#5B6FA6",
    "text": "#2C2C2C",
    "muted": "#7A7A7A",
    "header_bg": "#2C2C2C",
}


# ── Global CSS ──────────────────────────────────────────────────────────────
st.markdown(
    f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

  html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    background-color: {COLOURS['bg']};
    color: {COLOURS['text']};
  }}

  h1, h2, h3 {{
    font-family: 'DM Serif Display', serif;
  }}

  section[data-testid="stSidebar"] {{
    background-color: {COLOURS['header_bg']};
    color: white;
  }}

  section[data-testid="stSidebar"] * {{
    color: white !important;
  }}

  section[data-testid="stSidebar"] .stSelectbox > div > div,
  section[data-testid="stSidebar"] .stNumberInput input,
  section[data-testid="stSidebar"] .stMultiSelect > div > div {{
    background-color: #3d3d3d;
    border: 1px solid #555;
  }}

  .recipe-card {{
    background: {COLOURS['card']};
    border-radius: 10px;
    padding: 14px 16px;
    margin: 6px 4px;
    border-left: 5px solid;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    transition: box-shadow 0.15s;
  }}

  .recipe-card:hover {{
    box-shadow: 0 4px 14px rgba(0,0,0,0.14);
  }}

  .recipe-card.tried {{
    border-color: {COLOURS['Tried']};
  }}

  .recipe-card.not-tried {{
    border-color: {COLOURS['Not Tried']};
  }}

  .badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-top: 4px;
  }}

  .badge-tried {{
    background: {COLOURS['Tried']}22;
    color: {COLOURS['Tried']};
  }}

  .badge-nottried {{
    background: {COLOURS['Not Tried']}22;
    color: {COLOURS['Not Tried']};
  }}

  .type-chip {{
    display: inline-block;
    padding: 2px 9px;
    border-radius: 12px;
    font-size: 11px;
    background: {COLOURS['accent']}18;
    color: {COLOURS['accent']};
    font-weight: 500;
    margin-left: 6px;
  }}

  .plan-row-header {{
    background: {COLOURS['header_bg']};
    color: white;
    border-radius: 8px 8px 0 0;
    padding: 10px 18px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 18px;
  }}

  .plan-row-body {{
    background: #F2EFE9;
    border-radius: 0 0 8px 8px;
    padding: 10px 14px 14px 14px;
    margin-bottom: 4px;
  }}

  .similarity-bar-bg {{
    background: #e0e0e0;
    border-radius: 4px;
    height: 6px;
    width: 100%;
    margin-top: 4px;
  }}

  .similarity-bar-fill {{
    background: {COLOURS['accent']};
    border-radius: 4px;
    height: 6px;
  }}

  div[data-testid="stExpander"] > details > summary {{
    font-family: 'DM Serif Display', serif;
    font-size: 17px;
  }}

  .legend-dot {{
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 5px;
    vertical-align: middle;
  }}

  .price-line {{
    margin-top: 6px;
    font-size: 12px;
    color: {COLOURS['muted']};
  }}

  .price-line b {{
    color: {COLOURS['text']};
  }}
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardise the recipe data."""

    df.columns = [str(c).strip() for c in df.columns]

    # Remove blank exported columns
    df = df.loc[:, [c for c in df.columns if str(c).strip() != ""]].copy()

    exact_map = {
        "name": "Name",
        "recipe name": "Name",
        "servings": "Servings",
        "serving": "Servings",
        "ingredients": "Ingredients",
        "ingredient": "Ingredients",
        "steps": "Steps",
        "method": "Steps",
        "tried": "Tried",
        "type": "Type",
    }

    rename_map = {}

    for c in df.columns:
        lc = str(c).strip().lower()

        if lc in exact_map:
            new_name = exact_map[lc]

            # Only allow one source column for each final column
            if new_name not in rename_map.values():
                rename_map[c] = new_name

    df = df.rename(columns=rename_map)

    # Safety: if duplicate columns still exist, keep the first one
    df = df.loc[:, ~df.columns.duplicated()].copy()

    required = ["Name", "Servings", "Ingredients", "Steps", "Tried", "Type"]

    for col in required:
        if col not in df.columns:
            df[col] = ""

    df["Name"] = df["Name"].fillna("").astype(str).str.strip()
    df["Servings"] = pd.to_numeric(df["Servings"], errors="coerce").fillna(1).astype(int)
    df["Ingredients"] = df["Ingredients"].fillna("").astype(str)
    df["Steps"] = df["Steps"].fillna("").astype(str)
    df["Tried"] = df["Tried"].fillna("Not Tried").astype(str).str.strip()
    df["Type"] = df["Type"].fillna("Other").astype(str).str.strip()

    df = df[df["Name"] != ""]

    return df[required].reset_index(drop=True)

def parse_price(value):
    """Convert £0.39, 0.39, N/A, blanks etc into a float or NaN."""

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value == "" or value.lower() in {"n/a", "na", "nan", "none", "-"}:
        return np.nan

    value = value.replace("£", "").replace(",", "").strip()

    try:
        return float(value)
    except ValueError:
        return np.nan


def clean_price_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the price sheet."""

    df.columns = [str(c).strip() for c in df.columns]

    ingredient_col = None

    for col in df.columns:
        if str(col).lower() in {"ingredient", "ingredients"}:
            ingredient_col = col
            break

    if ingredient_col is None:
        return pd.DataFrame()

    df = df.rename(columns={ingredient_col: "Ingredients"})

    if "Unit" not in df.columns:
        df["Unit"] = "piece"

    if "Count" not in df.columns:
        df["Count"] = 0

    df["Ingredients"] = df["Ingredients"].fillna("").astype(str).str.strip()
    df["Unit"] = df["Unit"].fillna("piece").astype(str).str.strip().str.lower()

    df = df[df["Ingredients"] != ""]

    supermarket_cols = [
        c for c in df.columns
        if c not in {"Ingredients", "Count", "Unit"}
    ]

    for col in supermarket_cols:
        df[col] = df[col].apply(parse_price)

    df["Price_Key"] = df["Ingredients"].apply(normalise_ingredient_token)

    # Count is kept only as reference data. It is not used in the price calculation.
    # If duplicate ingredient keys exist, keep the first one in the sheet.
    df = df.drop_duplicates(subset=["Price_Key"], keep="first")

    return df.reset_index(drop=True)

@st.cache_data(ttl=300)
def load_recipes() -> pd.DataFrame:
    """Load recipe data from the published Google Sheet CSV link."""

    raw_df = pd.read_csv(GOOGLE_SHEET_CSV_URL)

    return clean_df(raw_df)


@st.cache_data(ttl=300)
def load_prices() -> pd.DataFrame:
    """Load price data from the published Price sheet CSV link."""

    if not PRICE_SHEET_CSV_URL or "PASTE_PRICE_SHEET_CSV_LINK_HERE" in PRICE_SHEET_CSV_URL:
        return pd.DataFrame()

    raw_df = pd.read_csv(PRICE_SHEET_CSV_URL)

    return clean_price_df(raw_df)


# ══════════════════════════════════════════════════════════════════════════════
# INGREDIENT SCALING
# ══════════════════════════════════════════════════════════════════════════════

def _parse_number(token: str):
    """Try to parse a leading numeric value."""

    token = str(token).strip()

    mixed = re.match(r"^(\d+)\s+(\d+)\s*/\s*(\d+)", token)
    if mixed:
        value = int(mixed.group(1)) + float(mixed.group(2)) / float(mixed.group(3))
        return value, mixed.end()

    frac = re.match(r"^(\d+)\s*/\s*(\d+)", token)
    if frac:
        value = float(frac.group(1)) / float(frac.group(2))
        return value, frac.end()

    num = re.match(r"^(\d+\.?\d*)", token)
    if num:
        return float(num.group(1)), num.end()

    return None, 0


def scale_ingredients(
    ingredients_str: str,
    original_servings: int,
    desired_servings: int,
) -> list[str]:
    """Scale a semicolon-separated ingredient string."""

    if original_servings == 0:
        original_servings = 1

    ratio = desired_servings / original_servings

    items = [i.strip() for i in str(ingredients_str).split(";") if i.strip()]
    scaled = []

    for item in items:
        value, end_idx = _parse_number(item)

        if value is not None:
            new_val = value * ratio

            if new_val == int(new_val):
                formatted = str(int(new_val))
            else:
                formatted = f"{new_val:.2f}".rstrip("0").rstrip(".")

            scaled.append(formatted + item[end_idx:])
        else:
            scaled.append(item)

    return scaled


# ══════════════════════════════════════════════════════════════════════════════
# SIMILARITY, INGREDIENTS AND PRICE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def compute_similarity_matrix(ingredients_list: tuple) -> np.ndarray:
    """TF-IDF cosine similarity across all recipe ingredient strings."""

    vectoriser = TfidfVectorizer(
        tokenizer=lambda x: [
            t.strip().lower()
            for t in re.split(r"[;,\d\s]+", x)
            if len(t.strip()) > 2
        ],
        token_pattern=None,
    )

    tfidf = vectoriser.fit_transform(list(ingredients_list))
    return cosine_similarity(tfidf)


def normalise_ingredient_token(token: str) -> str:
    """Clean ingredient text so similar ingredient names match better."""

    token = str(token).lower().strip()

    token = re.sub(r"\d+", "", token)
    token = re.sub(r"[^\w\s]", " ", token)
    token = re.sub(r"\s+", " ", token).strip()

    stop_words = {
        "g", "kg", "ml", "l", "tbsp", "tsp", "tablespoon", "tablespoons",
        "teaspoon", "teaspoons", "cup", "cups", "pinch", "handful",
        "small", "medium", "large", "fresh", "dried", "chopped", "sliced",
        "diced", "minced", "grated", "optional", "to", "taste", "of",
        "and", "or", "a", "an", "the", "clove", "cloves",
    }

    words = [w for w in token.split() if w not in stop_words and len(w) > 2]

    replacements = {
        "chickens": "chicken",
        "breasts": "breast",
        "thighs": "thigh",
        "onions": "onion",
        "tomatoes": "tomato",
        "potatoes": "potato",
        "peppers": "pepper",
        "noodles": "noodle",
        "eggs": "egg",
        "garlics": "garlic",
    }

    words = [replacements.get(w, w) for w in words]

    if "garlic" in words:
        return "garlic"

    # Remove duplicate words while preserving order.
    words = list(dict.fromkeys(words))

    return " ".join(words).strip()


def ingredient_tokens(ingredients_str: str) -> list[str]:
    """Convert a recipe's ingredient cell into cleaned ingredient tokens."""

    raw_items = [i.strip() for i in str(ingredients_str).split(";") if i.strip()]
    cleaned = []

    for item in raw_items:
        token = normalise_ingredient_token(item)

        if token:
            cleaned.append(token)

    return cleaned


def supermarket_columns(price_df: pd.DataFrame) -> list[str]:
    """Find supermarket columns in the price table."""

    if price_df.empty:
        return []

    return [
        c for c in price_df.columns
        if c not in {"Ingredients", "Count", "Unit", "Price_Key"}
    ]


def empty_price_summary(price_df: pd.DataFrame) -> dict:
    """Create an empty price summary."""

    shops = supermarket_columns(price_df)

    return {
        "totals": {shop: 0.0 for shop in shops},
        "missing": {shop: 0 for shop in shops},
        "best_shop": None,
        "best_total": np.nan,
        "best_missing": 0,
        "has_prices": False,
    }

def parse_ingredient_quantity(item: str) -> tuple[float, str | None, str]:
    """
    Read the amount and unit from an ingredient line.

    Examples:
    - 200g sugar      -> 200, g
    - 1.5 tsp oil    -> 1.5, tsp
    - 2 egg          -> 2, piece
    - onion          -> 1, piece
    """

    item = str(item).strip()

    value, end_idx = _parse_number(item)

    if value is None:
        return 1.0, "piece", item

    rest = item[end_idx:].strip()

    unit_match = re.match(r"^([a-zA-Z]+)", rest)

    if unit_match:
        unit = unit_match.group(1).lower()
        ingredient_text = rest[unit_match.end():].strip()
    else:
        unit = "piece"
        ingredient_text = rest.strip()

    return float(value), unit, ingredient_text


def convert_recipe_amount_to_price_units(amount: float, recipe_unit: str | None, price_unit: str) -> float | None:
    """
    Convert a recipe amount into the unit used by the Price sheet.

    Price sheet units supported:
    - 100g
    - 100ml
    - piece
    """

    recipe_unit = str(recipe_unit or "piece").lower().strip()
    price_unit = str(price_unit or "piece").lower().strip()

    weight_units = {
        "g": 1,
        "gram": 1,
        "grams": 1,
        "kg": 1000,
        "kilogram": 1000,
        "kilograms": 1000,
    }

    volume_units = {
        "ml": 1,
        "millilitre": 1,
        "millilitres": 1,
        "l": 1000,
        "litre": 1000,
        "litres": 1000,
    }

    piece_units = {
        "piece",
        "pieces",
        "whole",
        "egg",
        "eggs",
        "onion",
        "onions",
        "clove",
        "cloves",
    }

    if price_unit == "100g":
        if recipe_unit in weight_units:
            grams = amount * weight_units[recipe_unit]
            return grams / 100

        # If the recipe does not give a weight, assume one 100g price unit.
        return 1.0

    if price_unit == "100ml":
        if recipe_unit in volume_units:
            ml = amount * volume_units[recipe_unit]
            return ml / 100

        # If the recipe does not give a volume, assume one 100ml price unit.
        return 1.0

    if price_unit in {"piece", "each", "1 piece"}:
        if recipe_unit in piece_units:
            return amount

        # If it says "2 large onion", this still returns 2.
        return amount

    return None


def ingredient_items(ingredients_str: str) -> list[str]:
    """Split a recipe ingredient cell into ingredient lines."""

    return [i.strip() for i in str(ingredients_str).split(";") if i.strip()]



def calculate_tokens_price(
    tokens: list[str],
    price_df: pd.DataFrame,
    multiplier: float = 1.0,
) -> dict:
    """
    Old fallback price calculator.

    This is kept so the app does not break if anything else calls it,
    but recipe pricing now uses calculate_recipe_price(), which reads
    amounts and units properly.
    """

    if price_df.empty:
        return empty_price_summary(price_df)

    shops = supermarket_columns(price_df)

    totals = {shop: 0.0 for shop in shops}
    missing = {shop: 0 for shop in shops}

    price_lookup = price_df.set_index("Price_Key")

    for token in tokens:
        key = normalise_ingredient_token(token)

        if not key or key not in price_lookup.index:
            for shop in shops:
                missing[shop] += 1
            continue

        row = price_lookup.loc[key]
        price_unit = row.get("Unit", "piece")
        quantity_units = convert_recipe_amount_to_price_units(1.0, "piece", price_unit)

        if quantity_units is None:
            quantity_units = 1.0

        for shop in shops:
            price = row[shop]

            if pd.isna(price):
                missing[shop] += 1
            else:
                totals[shop] += float(price) * quantity_units * multiplier

    if not shops:
        return empty_price_summary(price_df)

    best_shop = sorted(
        shops,
        key=lambda shop: (
            missing[shop],
            totals[shop],
        ),
    )[0]

    return {
        "totals": totals,
        "missing": missing,
        "best_shop": best_shop,
        "best_total": totals[best_shop],
        "best_missing": missing[best_shop],
        "has_prices": True,
    }



def calculate_recipe_price(row: pd.Series, serving_override: int, price_df: pd.DataFrame) -> dict:
    """Calculate the best shop and price for one recipe using ingredient quantities."""

    if price_df.empty:
        return empty_price_summary(price_df)

    shops = supermarket_columns(price_df)

    totals = {shop: 0.0 for shop in shops}
    missing = {shop: 0 for shop in shops}

    price_lookup = price_df.set_index("Price_Key")

    original_servings = int(row["Servings"]) if int(row["Servings"]) > 0 else 1
    serving_multiplier = serving_override / original_servings

    for item in ingredient_items(row["Ingredients"]):
        amount, recipe_unit, ingredient_text = parse_ingredient_quantity(item)

        key = normalise_ingredient_token(ingredient_text)

        if not key or key not in price_lookup.index:
            for shop in shops:
                missing[shop] += 1
            continue

        price_row = price_lookup.loc[key]
        price_unit = price_row.get("Unit", "piece")

        quantity_units = convert_recipe_amount_to_price_units(
            amount=amount,
            recipe_unit=recipe_unit,
            price_unit=price_unit,
        )

        if quantity_units is None:
            for shop in shops:
                missing[shop] += 1
            continue

        quantity_units = quantity_units * serving_multiplier

        for shop in shops:
            price = price_row[shop]

            if pd.isna(price):
                missing[shop] += 1
            else:
                totals[shop] += float(price) * quantity_units

    if not shops:
        return empty_price_summary(price_df)

    best_shop = sorted(
        shops,
        key=lambda shop: (
            missing[shop],
            totals[shop],
        ),
    )[0]

    return {
        "totals": totals,
        "missing": missing,
        "best_shop": best_shop,
        "best_total": totals[best_shop],
        "best_missing": missing[best_shop],
        "has_prices": True,
    }




def combine_price_summaries(price_summaries: list[dict], price_df: pd.DataFrame) -> dict:
    """Combine recipe price summaries into one meal-plan summary."""

    if price_df.empty:
        return empty_price_summary(price_df)

    shops = supermarket_columns(price_df)

    totals = {shop: 0.0 for shop in shops}
    missing = {shop: 0 for shop in shops}

    for summary in price_summaries:
        for shop in shops:
            totals[shop] += summary["totals"].get(shop, 0.0)
            missing[shop] += summary["missing"].get(shop, 0)

    if not shops:
        return empty_price_summary(price_df)

    best_shop = sorted(
        shops,
        key=lambda shop: (
            missing[shop],
            totals[shop],
        ),
    )[0]

    return {
        "totals": totals,
        "missing": missing,
        "best_shop": best_shop,
        "best_total": totals[best_shop],
        "best_missing": missing[best_shop],
        "has_prices": True,
    }

def is_pasta_recipe(name: str) -> bool:
    """Return True if the recipe title contains Pasta."""

    return "pasta" in str(name).lower()

def format_price(value) -> str:
    """Format a price value."""

    if pd.isna(value):
        return "N/A"

    return f"£{float(value):.2f}"


def format_shop_price(summary: dict) -> str:
    """Format best shop and price."""

    if not summary or not summary.get("has_prices") or summary.get("best_shop") is None:
        return "No price data"

    missing = int(summary.get("best_missing", 0))

    if missing == 0:
        return (
            f"{summary['best_shop']} "
            f"{format_price(summary['best_total'])}"
        )

    return (
        f"{summary['best_shop']} "
        f"{format_price(summary['best_total'])} "
        f"({missing} N/A)"
    )


def format_all_shop_prices(summary: dict) -> str:
    """Format all supermarket prices for display under each meal plan."""

    if not summary or not summary.get("has_prices"):
        return "No price data loaded."

    parts = []

    for shop, total in summary["totals"].items():
        missing = int(summary["missing"].get(shop, 0))

        if missing == 0:
            parts.append(f"**{shop}:** {format_price(total)}")
        else:
            parts.append(f"**{shop}:** {format_price(total)} ({missing} N/A)")

    return " &nbsp; | &nbsp; ".join(parts)
    

def plan_ingredient_coverage_score(plan_recipes: pd.DataFrame) -> tuple[float, dict]:
    """Score a meal plan based on how many ingredient tokens are shared."""

    all_tokens = []

    for _, row in plan_recipes.iterrows():
        all_tokens.extend(ingredient_tokens(row["Ingredients"]))

    total_ingredients = len(all_tokens)

    if total_ingredients == 0:
        return 0.0, {
            "total_ingredients": 0,
            "shared_ingredients": 0,
            "unique_ingredients": 0,
            "repeated_ingredient_names": [],
            "unmatched_ingredient_names": [],
        }

    counts = pd.Series(all_tokens).value_counts()

    shared_ingredient_count = int(counts[counts >= 2].sum())
    unique_ingredient_count = int(counts[counts == 1].sum())

    score = shared_ingredient_count / total_ingredients

    details = {
        "total_ingredients": int(total_ingredients),
        "shared_ingredients": shared_ingredient_count,
        "unique_ingredients": unique_ingredient_count,
        "repeated_ingredient_names": counts[counts >= 2].index.tolist(),
        "unmatched_ingredient_names": counts[counts == 1].index.tolist(),
    }

    return float(score), details


# ══════════════════════════════════════════════════════════════════════════════
# MEAL PLAN GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_meal_plan(
    df: pd.DataFrame,
    n_meals: int,
    n_desserts: int,
    n_drinks: int,
    excluded_types: list[str],
    excluded_recipes: list[str],
    max_pasta_per_plan: int,
    sim_matrix: np.ndarray,
) -> tuple[list[int], float]:
    """Generate one meal plan."""

    type_col = df["Type"].str.lower()

    dessert_mask = type_col.str.contains("dessert|cake|bake|sweet|pudding|cookie|brownie", na=False)
    drink_mask = type_col.str.contains("drink|smoothie|juice|shake", na=False)

    if excluded_types:
        excluded_lower = [e.lower() for e in excluded_types]
        excluded_type_mask = type_col.isin(excluded_lower)
    else:
        excluded_type_mask = pd.Series([False] * len(df), index=df.index)

    if excluded_recipes:
        excluded_recipe_mask = df["Name"].isin(excluded_recipes)
    else:
        excluded_recipe_mask = pd.Series([False] * len(df), index=df.index)

    excluded_mask = excluded_type_mask | excluded_recipe_mask

    dessert_idx = df.index[dessert_mask & ~excluded_mask].tolist()
    drink_idx = df.index[drink_mask & ~excluded_mask].tolist()
    main_idx = df.index[~dessert_mask & ~drink_mask & ~excluded_mask].tolist()

    selected = []

    selected.extend(dessert_idx[:n_desserts])
    selected.extend(drink_idx[:n_drinks])

    n_main = n_meals - len(selected)

    if n_main < 0:
        n_main = 0

    def pasta_count(indices):
        return sum(is_pasta_recipe(df.loc[i, "Name"]) for i in indices)

    def can_add_recipe(current_indices, candidate_index):
        if not is_pasta_recipe(df.loc[candidate_index, "Name"]):
            return True

        return pasta_count(current_indices) < max_pasta_per_plan

    if len(main_idx) >= 1 and n_main >= 1:
        allowed_main_idx = [
            i for i in main_idx
            if can_add_recipe(selected, i)
        ]

        if n_main == 1:
            if allowed_main_idx:
                selected.append(allowed_main_idx[0])

        else:
            possible_pairs = []

            for a, b in combinations(main_idx[:30], 2):
                test_indices = selected + [a, b]

                if pasta_count(test_indices) <= max_pasta_per_plan:
                    possible_pairs.append((a, b))

            if possible_pairs:
                best_pair = possible_pairs[0]
                best_sim = -1

                for a, b in possible_pairs:
                    s = sim_matrix[a, b]

                    if s > best_sim:
                        best_sim = s
                        best_pair = (a, b)

                chosen_main = list(best_pair)
            else:
                chosen_main = []

                for candidate in main_idx:
                    if len(chosen_main) >= n_main:
                        break

                    if can_add_recipe(selected + chosen_main, candidate):
                        chosen_main.append(candidate)

            remaining = [i for i in main_idx if i not in chosen_main]

            while len(chosen_main) < n_main and remaining:
                allowed_remaining = [
                    r for r in remaining
                    if can_add_recipe(selected + chosen_main, r)
                ]

                if not allowed_remaining:
                    break

                if chosen_main:
                    avg_sims = [
                        np.mean([sim_matrix[r, c] for c in chosen_main])
                        for r in allowed_remaining
                    ]

                    best_next = allowed_remaining[int(np.argmax(avg_sims))]
                else:
                    best_next = allowed_remaining[0]

                chosen_main.append(best_next)
                remaining.remove(best_next)

            selected.extend(chosen_main)

    selected = list(dict.fromkeys(selected))

    if len(selected) >= 1:
        plan_recipes = df.iloc[selected].reset_index(drop=True)
        coverage_score, _ = plan_ingredient_coverage_score(plan_recipes)
    else:
        coverage_score = 0.0

    return selected, coverage_score



def build_multiple_plans(
    df: pd.DataFrame,
    n_plans: int,
    n_meals: int,
    n_desserts: int,
    n_drinks: int,
    excluded_types: list[str],
    excluded_recipes: list[str],
    max_pasta_per_plan: int,
    sim_matrix: np.ndarray,
) -> list[tuple[list[int], float]]:
    """Build multiple meal plans."""

    plans = []
    used_counts = np.zeros(len(df))

    for _ in range(n_plans):
        adj_sim = sim_matrix.copy()

        for i in range(len(df)):
            if used_counts[i] > 0:
                adj_sim[i, :] *= 0.5 ** used_counts[i]
                adj_sim[:, i] *= 0.5 ** used_counts[i]

        indices, score = generate_meal_plan(
            df,
            n_meals,
            n_desserts,
            n_drinks,
            excluded_types,
            excluded_recipes,
            max_pasta_per_plan,
            adj_sim,
        )

        plans.append((indices, score))

        for i in indices:
            used_counts[i] += 1

    plans.sort(key=lambda x: x[1], reverse=True)

    return plans

# ══════════════════════════════════════════════════════════════════════════════
# PDF EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def safe_pdf_text(value):
    """Make text safe for PDF output."""

    value = str(value)
    value = value.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value.encode("latin-1", "replace").decode("latin-1")


def break_long_words(text, max_len=45):
    """Break very long chunks so FPDF does not crash."""

    words = str(text).split(" ")
    fixed_words = []

    for word in words:
        if len(word) > max_len:
            fixed_words.extend(textwrap.wrap(word, max_len))
        else:
            fixed_words.append(word)

    return " ".join(fixed_words)


def pdf_line(value):
    """Clean text and prevent FPDF line-width errors."""

    return break_long_words(safe_pdf_text(value))


def generate_pdf(plan_recipes: pd.DataFrame, serving_overrides: dict, plan_num: int) -> bytes:
    """Generate a PDF of the meal plan."""

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(left=15, top=15, right=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, f"Meal Plan {plan_num}", ln=True, align="C")

    pdf.ln(4)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"{len(plan_recipes)} recipes", ln=True, align="C")
    pdf.set_text_color(0, 0, 0)

    pdf.ln(6)

    for _, row in plan_recipes.iterrows():
        desired = serving_overrides.get(row["Name"], row["Servings"])
        scaled_ing = scale_ingredients(row["Ingredients"], row["Servings"], desired)
        steps = [s.strip() for s in str(row["Steps"]).split(";") if s.strip()]

        pdf.set_x(pdf.l_margin)
        pdf.set_fill_color(44, 44, 44)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 9, pdf_line(row["Name"]), fill=True)

        pdf.set_text_color(0, 0, 0)

        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(120, 120, 120)

        tried_label = row["Tried"] if row["Tried"] else "Not Tried"
        meta = f"{row['Type']} | Serves {desired} | {tried_label}"

        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 6, pdf_line(meta))

        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 10)
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 7, "Ingredients", ln=True)

        pdf.set_font("Helvetica", "", 9)

        for ing in scaled_ing:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5, "- " + pdf_line(ing))

        pdf.ln(3)

        pdf.set_font("Helvetica", "B", 10)
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 7, "Method", ln=True)

        pdf.set_font("Helvetica", "", 9)

        for i, step in enumerate(steps, 1):
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5, f"{i}. {pdf_line(step)}")

        pdf.ln(6)

    return bytes(pdf.output())


# ══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def render_recipe_card(
    row: pd.Series,
    serving_override: int,
    key_prefix: str,
    price_summary: dict | None = None,
):
    """Render a recipe box using a Streamlit expander."""

    tried_value = str(row["Tried"]).lower()

    tried_class = "tried" if tried_value == "tried" else "not-tried"
    tried_badge = "badge-tried" if tried_value == "tried" else "badge-nottried"

    scaled = scale_ingredients(row["Ingredients"], row["Servings"], serving_override)
    steps = [s.strip() for s in str(row["Steps"]).split(";") if s.strip()]
    n_ing = len(scaled)

    if price_summary and price_summary.get("has_prices"):
        price_text = format_shop_price(price_summary)
        missing_count = int(price_summary.get("best_missing", 0))

        if missing_count == 0:
            missing_text = "All ingredients priced"
        elif missing_count == 1:
            missing_text = "1 ingredient missing price"
        else:
            missing_text = f"{missing_count} ingredients missing prices"
    else:
        price_text = "No price data"
        missing_text = "Price data unavailable"

    st.markdown(
        f"""
    <div class="recipe-card {tried_class}">
      <div style="font-family:'DM Serif Display',serif;font-size:15px;font-weight:600;color:{COLOURS['text']}">
        {row['Name']}
      </div>
      <div style="margin-top:4px;">
        <span class="badge {tried_badge}">{row['Tried']}</span>
        <span class="type-chip">{row['Type']}</span>
      </div>
      <div style="margin-top:8px;font-size:12px;color:{COLOURS['muted']}">
        Serves <b style="color:{COLOURS['text']}">{serving_override}</b>
        &nbsp;·&nbsp;
        {n_ing} ingredients
      </div>
      <div class="price-line">
        Best price: <b>{price_text}</b>
      </div>
      <div style="margin-top:2px;font-size:10.5px;color:{COLOURS['muted']};font-style:italic;">
        {missing_text}
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.expander("Open recipe"):
        st.caption(
            f"Serves: {serving_override}  |  "
            f"Type: {row['Type']}  |  "
            f"Status: {row['Tried']}  |  "
            f"Best price: {price_text}  |  "
            f"{missing_text}"
        )

        st.markdown("---")

        col_i, col_s = st.columns([1, 1])

        with col_i:
            st.markdown("**Ingredients**")

            for ing in scaled:
                st.markdown(f"- {ing}")

        with col_s:
            st.markdown("**Method**")

            for i, step in enumerate(steps, 1):
                st.markdown(f"**{i}.** {step}")
# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.markdown(
        f"""
    <div style="background:{COLOURS['header_bg']};padding:28px 32px 22px;border-radius:12px;margin-bottom:24px;">
      <h1 style="color:white;margin:0;font-family:'DM Serif Display',serif;">🍽️ Meal Planner</h1>
      <p style="color:#aaa;margin:6px 0 0;font-size:14px;">
        Generate meal plans from your recipe list
      </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    try:
        df = load_recipes()
    except Exception as e:
        st.error("Could not load recipes from the published Google Sheet CSV.")
        st.exception(e)
        return

    try:
        price_df = load_prices()
    except Exception as e:
        price_df = pd.DataFrame()
        st.warning("Recipes loaded, but the Price sheet could not be loaded.")
        st.exception(e)

    if df.empty:
        st.warning("Your Google Sheet loaded, but it does not contain any recipes.")
        return

    # Sidebar
    st.sidebar.markdown("## Welcome")
    st.sidebar.write("Choose your settings, then generate meal plans.")

    if st.sidebar.button("Refresh Google Sheet data"):
        st.cache_data.clear()

        for key in ["plans", "sim"]:
            if key in st.session_state:
                del st.session_state[key]

        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Plan Settings")

    all_types = sorted(df["Type"].dropna().unique().tolist())

    n_plans = st.sidebar.number_input(
        "Number of meal plans to generate",
        min_value=1,
        max_value=10,
        value=3,
    )

    n_meals = st.sidebar.number_input(
        "Recipes per plan",
        min_value=1,
        max_value=10,
        value=4,
    )

    n_desserts = st.sidebar.number_input(
        "Desserts per plan",
        min_value=0,
        max_value=5,
        value=1,
    )

    n_drinks = st.sidebar.number_input(
        "Drinks per plan",
        min_value=0,
        max_value=5,
        value=0,
    )

    excluded_types = st.sidebar.multiselect(
        "Exclude recipe types",
        options=all_types,
        default=[],
    )

    excluded_recipes = st.sidebar.multiselect(
        "Exclude specific recipes",
        options=sorted(df["Name"].dropna().unique().tolist()),
        default=[],
    )

    max_pasta_per_plan = st.sidebar.number_input(
        "Max pasta dishes per plan",
        min_value=0,
        max_value=10,
        value=1,
    )

    sort_by = st.sidebar.selectbox(
        "Sort meal plans by",
        [
            "Shared ingredient score",
            "Cheapest meal plan",
        ],
    )

    # Servings
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Servings")

    default_main_serving = st.sidebar.number_input(
        "Default servings for meals",
        min_value=1,
        max_value=20,
        value=2,
    )

    default_drink_serving = st.sidebar.number_input(
        "Default servings for drinks",
        min_value=1,
        max_value=20,
        value=1,
    )

    if "serving_overrides" not in st.session_state:
        st.session_state["serving_overrides"] = {}

    def recipe_category(recipe_type):
        recipe_type = str(recipe_type).lower()

        if re.search(r"dessert|cake|bake|sweet|pudding|cookie|brownie", recipe_type):
            return "Dessert"

        if re.search(r"drink|smoothie|juice|shake", recipe_type):
            return "Drink"

        return "Meal"

    def default_serving_for_type(recipe_type):
        category = recipe_category(recipe_type)

        if category == "Drink":
            return default_drink_serving

        return default_main_serving

    meal_rows = df[df["Type"].apply(recipe_category) == "Meal"]
    dessert_rows = df[df["Type"].apply(recipe_category) == "Dessert"]
    drink_rows = df[df["Type"].apply(recipe_category) == "Drink"]

    with st.sidebar.expander("Meal serving overrides"):
        if meal_rows.empty:
            st.caption("No meal recipes found.")
        else:
            for _, row in meal_rows.iterrows():
                key = f"srv_{row['Name']}"
                default_value = default_serving_for_type(row["Type"])

                current = st.session_state["serving_overrides"].get(
                    row["Name"],
                    default_value,
                )

                new_val = st.number_input(
                    row["Name"],
                    min_value=1,
                    max_value=20,
                    value=int(current),
                    key=key,
                )

                st.session_state["serving_overrides"][row["Name"]] = new_val

    with st.sidebar.expander("Dessert servings"):
        if dessert_rows.empty:
            st.caption("No dessert recipes found.")
        else:
            st.caption("Desserts keep their original recipe servings.")

            for _, row in dessert_rows.iterrows():
                st.write(f"{row['Name']} — serves {row['Servings']}")

    with st.sidebar.expander("Drink serving overrides"):
        if drink_rows.empty:
            st.caption("No drink recipes found.")
        else:
            for _, row in drink_rows.iterrows():
                key = f"srv_{row['Name']}"
                default_value = default_serving_for_type(row["Type"])

                current = st.session_state["serving_overrides"].get(
                    row["Name"],
                    default_value,
                )

                new_val = st.number_input(
                    row["Name"],
                    min_value=1,
                    max_value=20,
                    value=int(current),
                    key=key,
                )

                st.session_state["serving_overrides"][row["Name"]] = new_val

    def get_serving(name):
        recipe_row = df[df["Name"] == name]

        if recipe_row.empty:
            return default_main_serving

        recipe_type = recipe_row.iloc[0]["Type"]
        original_servings = int(recipe_row.iloc[0]["Servings"])

        if recipe_category(recipe_type) == "Dessert":
            return original_servings

        return st.session_state["serving_overrides"].get(
            name,
            default_serving_for_type(recipe_type),
        )

    if price_df.empty:
        st.caption(f"{len(df)} recipes loaded from Google Sheets. No price data loaded yet.")
    else:
        st.caption(
            f"{len(df)} recipes loaded from Google Sheets. "
            f"{len(price_df)} priced ingredients loaded from Price sheet."
        )

    if st.button("Generate Meal Plans", use_container_width=True):
        with st.spinner("Analysing ingredients and building meal plans..."):
            sim = compute_similarity_matrix(tuple(df["Ingredients"].tolist()))

            plans = build_multiple_plans(
                df=df,
                n_plans=int(n_plans),
                n_meals=int(n_meals),
                n_desserts=int(n_desserts),
                n_drinks=int(n_drinks),
                excluded_types=excluded_types,
                excluded_recipes=excluded_recipes,
                max_pasta_per_plan=int(max_pasta_per_plan),
                sim_matrix=sim,
            )

            st.session_state["plans"] = plans
            st.session_state["sim"] = sim

    if "plans" not in st.session_state:
        st.markdown(
            """
        <div style='text-align:center;padding:60px;color:#aaa;'>
          <div style='font-size:48px;'>🥘</div>
          <p style='font-size:16px;margin-top:12px;'>
            Use the sidebar settings, then click <strong>Generate Meal Plans</strong>.
          </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        return

    plans = st.session_state["plans"]

    # Build display data, including fresh scores and prices.
    plan_display = []

    for recipe_indices, score in plans:
        plan_recipes = df.iloc[recipe_indices].reset_index(drop=True)
        coverage_score, coverage_details = plan_ingredient_coverage_score(plan_recipes)

        recipe_price_summaries = {}

        for _, row in plan_recipes.iterrows():
            serving = get_serving(row["Name"])
            recipe_price_summaries[row["Name"]] = calculate_recipe_price(
                row,
                serving,
                price_df,
            )

        plan_price_summary = combine_price_summaries(
            list(recipe_price_summaries.values()),
            price_df,
        )

        plan_display.append(
            {
                "recipe_indices": recipe_indices,
                "plan_recipes": plan_recipes,
                "coverage_score": coverage_score,
                "coverage_details": coverage_details,
                "recipe_price_summaries": recipe_price_summaries,
                "plan_price_summary": plan_price_summary,
            }
        )

    if sort_by == "Cheapest meal plan":
        plan_display.sort(
            key=lambda plan: (
                plan["plan_price_summary"].get("best_missing", 999999),
                plan["plan_price_summary"].get("best_total", 999999),
            )
        )
    else:
        plan_display.sort(
            key=lambda plan: plan["coverage_score"],
            reverse=True,
        )

    st.markdown(
        f"""
    <div style='display:flex;gap:20px;align-items:center;margin-bottom:12px;'>
      <span><span class='legend-dot' style='background:{COLOURS["Tried"]}'></span>Tried</span>
      <span><span class='legend-dot' style='background:{COLOURS["Not Tried"]}'></span>Not Tried</span>
      <span style='color:{COLOURS["muted"]};font-size:12px;'>
        Plans are sorted by {sort_by.lower()}
      </span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    for plan_idx, plan in enumerate(plan_display):
        plan_recipes = plan["plan_recipes"]
        coverage_score = plan["coverage_score"]
        coverage_details = plan["coverage_details"]
        recipe_price_summaries = plan["recipe_price_summaries"]
        plan_price_summary = plan["plan_price_summary"]

        pct = int(coverage_score * 100)
        best_price_text = format_shop_price(plan_price_summary)

        st.markdown(
            f"""
        <div class='plan-row-header'>
          Plan {plan_idx + 1}
          <span style='font-weight:300;font-size:11px;margin-left:12px;opacity:0.7;'>
            Shared ingredient score
          </span>
          <span style='font-weight:700;font-size:12px;margin-left:4px;'>{pct}%</span>
          <span style='font-weight:300;font-size:11px;margin-left:18px;opacity:0.7;'>
            Best total price
          </span>
          <span style='font-weight:700;font-size:12px;margin-left:4px;'>{best_price_text}</span>
        </div>
        <div class='plan-row-body'>
          <div class='similarity-bar-bg'>
            <div class='similarity-bar-fill' style='width:{min(pct, 100)}%'></div>
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.caption(
            f"{coverage_details['shared_ingredients']} of "
            f"{coverage_details['total_ingredients']} ingredients are shared with another recipe "
            f"in this plan. {coverage_details['unique_ingredients']} ingredients only appear once."
        )

        st.markdown(
            format_all_shop_prices(plan_price_summary),
            unsafe_allow_html=True,
        )

        with st.expander("See shared and unmatched ingredients"):
            col_shared, col_unique = st.columns(2)

            with col_shared:
                st.markdown("**Shared ingredients**")

                if coverage_details["repeated_ingredient_names"]:
                    for ing in coverage_details["repeated_ingredient_names"]:
                        st.markdown(f"- {ing}")
                else:
                    st.write("No shared ingredients.")

            with col_unique:
                st.markdown("**Only used once**")

                if coverage_details["unmatched_ingredient_names"]:
                    for ing in coverage_details["unmatched_ingredient_names"]:
                        st.markdown(f"- {ing}")
                else:
                    st.write("No unmatched ingredients.")

        n_cols = min(len(plan_recipes), 4)
        cols = st.columns(n_cols)

        for col_idx, (_, row) in enumerate(plan_recipes.iterrows()):
            serving = get_serving(row["Name"])
            recipe_price_summary = recipe_price_summaries.get(row["Name"])

            with cols[col_idx % n_cols]:
                render_recipe_card(
                    row,
                    serving,
                    f"plan{plan_idx}_rec{col_idx}",
                    recipe_price_summary,
                )

        serving_dict = {
            name: get_serving(name)
            for name in df["Name"]
        }

        pdf_bytes = generate_pdf(
            plan_recipes,
            serving_dict,
            plan_idx + 1,
        )

        st.download_button(
            label=f"Download Plan {plan_idx + 1} as PDF",
            data=pdf_bytes,
            file_name=f"meal_plan_{plan_idx + 1}.pdf",
            mime="application/pdf",
            key=f"dl_plan_{plan_idx}",
        )

        st.markdown("<hr style='margin:18px 0;border-color:#ddd;'>", unsafe_allow_html=True)

    with st.expander("Browse all recipes"):
        search = st.text_input("Search by name or ingredient", "")

        filter_type = st.multiselect(
            "Filter by type",
            options=all_types,
            default=[],
        )

        filter_tried = st.selectbox(
            "Filter by tried status",
            ["All", "Tried", "Not Tried"],
        )

        mask = pd.Series([True] * len(df), index=df.index)

        if search:
            mask &= (
                df["Name"].str.contains(search, case=False, na=False)
                | df["Ingredients"].str.contains(search, case=False, na=False)
            )

        if filter_type:
            mask &= df["Type"].isin(filter_type)

        if filter_tried != "All":
            mask &= df["Tried"] == filter_tried

        filtered = df[mask].reset_index(drop=True)

        st.caption(f"{len(filtered)} recipes")

        cols = st.columns(3)

        for i, (_, row) in enumerate(filtered.iterrows()):
            serving = get_serving(row["Name"])
            recipe_price_summary = calculate_recipe_price(row, serving, price_df)

            with cols[i % 3]:
                render_recipe_card(
                    row,
                    serving,
                    f"browse_{i}",
                    recipe_price_summary,
                )


if __name__ == "__main__":
    main()
