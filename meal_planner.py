import streamlit as st
import pandas as pd
import numpy as np
import re
import math
import textwrap
import html

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

# Price sheet ("Price 2" tab: Ingredient | Count | Shop | Name | Size | Price)
# IMPORTANT: replace this with the published CSV link for your "Price 2" tab
# specifically (it will have a different gid to your old Price tab).
PRICE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQPVOLW-IYb4T3HojVWtCxgXw1wmXl4TQSU1QAuDRc9A-o0h36DOXS5Rp6YagT-E2YB6Z0P1tcaxOj5/pub?gid=211392906&single=true&output=csv"


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

  /* Fix: the typed search text inside multiselect/selectbox boxes was
     inheriting forced-white text on a white input background, making it
     invisible. Force a dark background on every nested input/value part. */
  section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
    background-color: #3d3d3d !important;
  }}

  section[data-testid="stSidebar"] div[data-baseweb="select"] input {{
    background-color: transparent !important;
    color: white !important;
    -webkit-text-fill-color: white !important;
  }}

  section[data-testid="stSidebar"] div[data-baseweb="tag"] {{
    background-color: {COLOURS['accent']} !important;
  }}

  /* The dropdown option list renders in a portal outside the sidebar, so it
     does NOT inherit the sidebar's forced-white rule above - but force
     readable colours explicitly anyway in case Streamlit changes this. */
  ul[data-baseweb="menu"],
  ul[role="listbox"] {{
    background-color: #FFFFFF !important;
  }}

  ul[data-baseweb="menu"] li,
  ul[role="listbox"] li,
  ul[data-baseweb="menu"] li *,
  ul[role="listbox"] li * {{
    color: {COLOURS['text']} !important;
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
# DATA LOADING - RECIPES
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


@st.cache_data(ttl=300)
def load_recipes() -> pd.DataFrame:
    """Load recipe data from the published Google Sheet CSV link."""

    raw_df = pd.read_csv(GOOGLE_SHEET_CSV_URL)

    return clean_df(raw_df)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING - PRICE 2 (Ingredient | Count | Shop | Name | Size | Price)
# ══════════════════════════════════════════════════════════════════════════════

# Base-unit conversions used to turn a Size cell into a (quantity, unit_type) pair.
# unit_type is always one of "g", "ml", "piece" after conversion.
SIZE_UNIT_CONVERSIONS = {
    "g": ("g", 1), "gram": ("g", 1), "grams": ("g", 1),
    "kg": ("g", 1000), "kilogram": ("g", 1000), "kilograms": ("g", 1000),
    "ml": ("ml", 1), "millilitre": ("ml", 1), "millilitres": ("ml", 1),
    "l": ("ml", 1000), "litre": ("ml", 1000), "litres": ("ml", 1000),
    "piece": ("piece", 1), "pieces": ("piece", 1),
}


def parse_size(value) -> tuple[float, str] | tuple[None, None]:
    """
    Parse a Size cell such as '400g', '1L', '1 Piece', or a multipack
    like '6x500ml', into (base_quantity, unit_type) where unit_type is
    'g', 'ml', or 'piece'. Returns (None, None) if it can't be parsed.
    """

    value = str(value).strip().lower()

    if not value:
        return None, None

    multipack = re.match(r"^(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*([a-z]+)$", value)

    if multipack:
        count = float(multipack.group(1))
        each = float(multipack.group(2))
        unit_word = multipack.group(3)

        if unit_word not in SIZE_UNIT_CONVERSIONS:
            return None, None

        base_unit, mult = SIZE_UNIT_CONVERSIONS[unit_word]
        return count * each * mult, base_unit

    single = re.match(r"^(\d+(?:\.\d+)?)\s*([a-z]+)$", value)

    if single:
        qty = float(single.group(1))
        unit_word = single.group(2)

        if unit_word not in SIZE_UNIT_CONVERSIONS:
            return None, None

        base_unit, mult = SIZE_UNIT_CONVERSIONS[unit_word]
        return qty * mult, base_unit

    return None, None


def strip_quantity_and_unit_from_ingredient(item: str) -> str:
    """
    Match the Google Sheets formula used to create the Price ingredient list.

    Removes:
    - leading number + recognised unit, e.g. 200g sugar -> sugar
    - leading number only, e.g. 2 chicken breast -> chicken breast
    - ranges, e.g. 2-3 tbsp oil -> oil
    """

    item = str(item).strip()

    item = re.sub(
        r"^\s*[0-9]+(?:\.[0-9]+)?(?:\s*-\s*[0-9]+(?:\.[0-9]+)?)?\s*"
        r"(?:g|kg|ml|litre|litres|l|tbsp|tsp|cm|piece)\b\s*"
        r"|^\s*[0-9]+(?:\.[0-9]+)?\s+",
        "",
        item,
        flags=re.IGNORECASE,
    )

    return item.strip()


def price_match_key(value: str) -> str:
    """
    Create the matching key used between recipe ingredients and the Price
    sheet's Ingredient column.

    - removes leading quantity + unit
    - removes leading quantity only
    - lowercases
    - removes punctuation
    - trims spaces
    """

    value = strip_quantity_and_unit_from_ingredient(value)

    value = str(value).strip().lower()
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def normalise_ingredient_token(token: str) -> str:
    """Clean ingredient text so similar ingredient names match better (used
    only for the shared-ingredient overlap score, not for pricing)."""

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

    # Remove duplicate words while preserving order.
    words = list(dict.fromkeys(words))

    return " ".join(words).strip()


def ingredient_items(ingredients_str: str) -> list[str]:
    """Split a recipe ingredient cell into ingredient lines."""

    return [i.strip() for i in str(ingredients_str).split(";") if i.strip()]


def ingredient_tokens(ingredients_str: str) -> list[str]:
    """Convert a recipe's ingredient cell into cleaned ingredient tokens
    (used for the shared-ingredient overlap score only)."""

    cleaned = []

    for item in ingredient_items(ingredients_str):
        ingredient_text = strip_quantity_and_unit_from_ingredient(item)
        token = normalise_ingredient_token(ingredient_text)

        if token:
            cleaned.append(token)

    return cleaned


@st.cache_data(ttl=300)
def load_prices() -> dict:
    """
    Load the Price 2 sheet (Ingredient | Count | Shop | Name | Size | Price)
    into a lookup structure:

    {
        "shops": ["Tesco", "ASDA", ...],
        "options": {
            "<price_match_key>": {
                "<shop>": [
                    {"name": ..., "size_qty": ..., "unit_type": "g"/"ml"/"piece", "price": ...},
                    ...
                ],
                ...
            },
            ...
        }
    }

    Multiple size options per ingredient per shop are all kept, so pricing
    can pick whichever adequate size is cheapest for the quantity needed.
    """

    if not PRICE_SHEET_CSV_URL or "PASTE_PRICE_SHEET_CSV_LINK_HERE" in PRICE_SHEET_CSV_URL:
        return {"shops": [], "options": {}}

    raw_df = pd.read_csv(PRICE_SHEET_CSV_URL)
    raw_df.columns = [str(c).strip() for c in raw_df.columns]

    colmap = {c.lower(): c for c in raw_df.columns}

    ingredient_col = colmap.get("ingredient")
    shop_col = colmap.get("shop")
    name_col = colmap.get("name")
    size_col = colmap.get("size")
    price_col = colmap.get("price")

    if not all([ingredient_col, shop_col, name_col, size_col, price_col]):
        return {"shops": [], "options": {}}

    options: dict = {}
    shops_seen: set = set()

    for _, row in raw_df.iterrows():
        ingredient_text = str(row[ingredient_col]).strip()

        if not ingredient_text:
            continue

        key = price_match_key(ingredient_text)

        if not key:
            continue

        shop = str(row[shop_col]).strip()

        if not shop:
            continue

        name = str(row[name_col]).strip()
        size_qty, unit_type = parse_size(row[size_col])
        price = parse_price(row[price_col])

        if size_qty is None or pd.isna(price):
            continue

        shops_seen.add(shop)

        options.setdefault(key, {}).setdefault(shop, []).append(
            {
                "name": name,
                "size_qty": size_qty,
                "unit_type": unit_type,
                "price": float(price),
            }
        )

    return {"shops": sorted(shops_seen), "options": options}


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
# SIMILARITY
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


def is_pasta_recipe(name: str) -> bool:
    """Return True if the recipe title contains Pasta."""

    return "pasta" in str(name).lower()


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
# PRICING (Price 2: whole-package, plan-level aggregation)
# ══════════════════════════════════════════════════════════════════════════════

WEIGHT_RECIPE_UNITS = {"g", "gram", "grams", "kg", "kilogram", "kilograms"}
VOLUME_RECIPE_UNITS = {"ml", "millilitre", "millilitres", "l", "litre", "litres"}

RECOGNISED_RECIPE_UNITS = {
    "g", "gram", "grams",
    "kg", "kilogram", "kilograms",
    "ml", "millilitre", "millilitres",
    "l", "litre", "litres",
    "tbsp", "tablespoon", "tablespoons",
    "tsp", "teaspoon", "teaspoons",
    "cup", "cups",
    "pinch", "pinches",
    "handful", "handfuls",
    "clove", "cloves",
    "piece", "pieces",
    "slice", "slices",
    "can", "cans",
    "tin", "tins",
    "jar", "jars",
    "packet", "packets",
    "pack", "packs",
    "cm",
}


def parse_ingredient_quantity(item: str) -> tuple[float, str | None, str]:
    """
    Read amount/unit for pricing, using the same cleaned ingredient name
    as the Google Sheets formula used to build the Price sheet.
    """

    item = str(item).strip()

    value, end_idx = _parse_number(item)

    if value is None:
        return 1.0, "piece", strip_quantity_and_unit_from_ingredient(item)

    rest = item[end_idx:].strip()

    unit_match = re.match(r"^([a-zA-Z]+)\b", rest)

    if unit_match:
        possible_unit = unit_match.group(1).lower()
        unit = possible_unit if possible_unit in RECOGNISED_RECIPE_UNITS else "piece"
    else:
        unit = "piece"

    ingredient_text = strip_quantity_and_unit_from_ingredient(item)

    if not ingredient_text:
        ingredient_text = rest.strip()

    return float(value), unit, ingredient_text


def recipe_amount_to_base(amount: float, recipe_unit: str | None) -> tuple[float, str]:
    """
    Convert a recipe amount + unit into a base quantity and unit_type
    ('g', 'ml', or 'piece') so it can be compared against Price sheet sizes.

    Anything that isn't a recognised weight or volume unit (tbsp, tsp, cup,
    pinch, handful, clove, slice, can, tin, jar, packet, pack, cm, piece, or
    unrecognised text) is treated as a countable 'piece' - i.e. the recipe's
    stated amount is the number of whole units needed.
    """

    recipe_unit = str(recipe_unit or "piece").lower().strip()

    if recipe_unit in WEIGHT_RECIPE_UNITS:
        mult = 1000 if recipe_unit in {"kg", "kilogram", "kilograms"} else 1
        return amount * mult, "g"

    if recipe_unit in VOLUME_RECIPE_UNITS:
        mult = 1000 if recipe_unit in {"l", "litre", "litres"} else 1
        return amount * mult, "ml"

    return amount, "piece"


def get_recipe_ingredient_requirements(row: pd.Series, serving_override: int) -> list[dict]:
    """
    Work out how much of each ingredient a recipe needs, scaled to the
    desired servings, in base units (g / ml / piece) ready for pricing.
    """

    original_servings = int(row["Servings"]) if int(row["Servings"]) > 0 else 1
    multiplier = serving_override / original_servings

    requirements = []

    for item in ingredient_items(row["Ingredients"]):
        amount, recipe_unit, ingredient_text = parse_ingredient_quantity(item)

        display_name = ingredient_text.strip()

        if not display_name:
            display_name = strip_quantity_and_unit_from_ingredient(item)

        if not display_name:
            display_name = item.strip()

        key = price_match_key(ingredient_text) if ingredient_text else price_match_key(item)
        base_qty, unit_type = recipe_amount_to_base(amount, recipe_unit)
        base_qty = base_qty * multiplier

        requirements.append(
            {
                "display_name": display_name,
                "key": key,
                "required_qty": base_qty,
                "unit_type": unit_type,
            }
        )

    return requirements


def format_quantity(qty: float, unit_type: str) -> str:
    """Format a base quantity for display, e.g. 1500 g -> '1.5kg'."""

    if qty is None:
        return ""

    if unit_type == "g":
        if qty >= 1000:
            return f"{qty / 1000:.2f}".rstrip("0").rstrip(".") + "kg"
        return f"{qty:.0f}g" if qty == int(qty) else f"{qty:.1f}g"

    if unit_type == "ml":
        if qty >= 1000:
            return f"{qty / 1000:.2f}".rstrip("0").rstrip(".") + "L"
        return f"{qty:.0f}ml" if qty == int(qty) else f"{qty:.1f}ml"

    # piece
    if qty == int(qty):
        n = int(qty)
        return f"{n} piece" + ("s" if n != 1 else "")

    return f"{qty:.2f} piece"


def best_option_for_shop(shop_options: list[dict], required_qty: float, unit_type: str) -> dict | None:
    """
    Pick the cheapest way to cover `required_qty` of `unit_type` from a
    shop's available product sizes for one ingredient.

    Only options whose unit_type matches are considered "adequate" candidates
    normally; for each, the number of whole packages needed is rounded UP
    (you can't buy 0.6 of a bottle of milk), and the cheapest total across
    all matching sizes is chosen (not necessarily the size numerically
    closest to what's needed - the cheapest way to cover it).

    If no option matches the required unit_type (e.g. recipe gives a weight
    but the product is only sold as a whole piece), the cheapest available
    option is used assuming a single pack, as a best-effort fallback.
    """

    if not shop_options:
        return None

    matching = [o for o in shop_options if o["unit_type"] == unit_type]
    candidates = matching if matching else shop_options

    best = None
    best_cost = None

    for opt in candidates:
        if matching and opt["size_qty"] > 0:
            packs = max(1, math.ceil(required_qty / opt["size_qty"]))
        else:
            # Cross-unit-type fallback, or a zero-size option: assume 1 pack.
            packs = 1

        cost = packs * opt["price"]

        if best_cost is None or cost < best_cost:
            best_cost = cost
            best = {
                "name": opt["name"],
                "size_qty": opt["size_qty"],
                "unit_type": opt["unit_type"],
                "price": opt["price"],
                "packs": packs,
                "total_price": cost,
                "approx": not bool(matching),
            }

    return best


def empty_price_result(shops: list[str] | None = None) -> dict:
    """A price result shape with no price data."""

    shops = shops or []

    return {
        "totals": {s: 0.0 for s in shops},
        "missing": {s: 0 for s in shops},
        "missing_items": {s: [] for s in shops},
        "ingredient_price_details": [],
        "items": [],
        "best_shop": None,
        "best_total": np.nan,
        "best_missing": 0,
        "best_missing_items": [],
        "has_prices": False,
    }


def calculate_recipe_price(row: pd.Series, serving_override: int, price_options: dict) -> dict:
    """
    Price ONE recipe in isolation - as if you were only buying ingredients
    for this recipe and nothing else. Each ingredient is rounded up to
    whole packages independently. This intentionally overstates cost when
    an ingredient is shared with other recipes in a plan (see
    build_plan_shopping_list for the plan-level aggregation that fixes
    that). Used for the per-recipe "Best price" shown on each card/PDF.
    """

    shops = price_options.get("shops", []) if price_options else []
    options = price_options.get("options", {}) if price_options else {}

    if not shops:
        return empty_price_result([])

    requirements = get_recipe_ingredient_requirements(row, serving_override)

    totals = {shop: 0.0 for shop in shops}
    missing = {shop: 0 for shop in shops}
    missing_items = {shop: [] for shop in shops}
    ingredient_price_details = []

    for req in requirements:
        key = req["key"]
        shop_options_for_key = options.get(key, {}) if key else {}

        item_missing = {}
        chosen = {}

        for shop in shops:
            shop_opts = shop_options_for_key.get(shop, [])
            best = best_option_for_shop(shop_opts, req["required_qty"], req["unit_type"])

            if best is None:
                missing[shop] += 1
                missing_items[shop].append(req["display_name"])
                item_missing[shop] = True
            else:
                totals[shop] += best["total_price"]
                chosen[shop] = best
                item_missing[shop] = False

        ingredient_price_details.append(
            {
                "display_name": req["display_name"],
                "key": key,
                "required_qty": req["required_qty"],
                "unit_type": req["unit_type"],
                "missing": item_missing,
                "chosen": chosen,
            }
        )

    best_shop = sorted(shops, key=lambda s: (missing[s], totals[s]))[0]

    return {
        "totals": totals,
        "missing": missing,
        "missing_items": missing_items,
        "ingredient_price_details": ingredient_price_details,
        "best_shop": best_shop,
        "best_total": totals[best_shop],
        "best_missing": missing[best_shop],
        "best_missing_items": missing_items[best_shop],
        "has_prices": True,
    }


def build_plan_shopping_list(
    plan_recipes: pd.DataFrame,
    get_serving_fn,
    price_options: dict,
) -> dict:
    """
    Price a WHOLE meal plan properly: sum each ingredient's raw required
    quantity across every recipe in the plan FIRST, then round up to whole
    packages ONCE per ingredient. This is what stops e.g. milk used a little
    in 4 recipes from being bought as 4 separate bottles - if the combined
    amount still fits in 1 bottle, only 1 is costed.

    Returns both the plan-level totals (for format_shop_price /
    format_all_shop_prices) and an itemised shopping list ("items") for the
    PDF and on-screen breakdown.
    """

    shops = price_options.get("shops", []) if price_options else []
    options = price_options.get("options", {}) if price_options else {}

    if not shops:
        return empty_price_result([])

    aggregated: dict = {}

    for _, row in plan_recipes.iterrows():
        serving = get_serving_fn(row["Name"])

        for req in get_recipe_ingredient_requirements(row, serving):
            key = req["key"]

            if not key:
                continue

            entry = aggregated.setdefault(
                key,
                {
                    "display_name": req["display_name"],
                    "unit_type": req["unit_type"],
                    "required_qty": 0.0,
                },
            )

            # Same key should always resolve to the same unit_type in
            # practice; if a mismatch ever occurs, keep the first seen type
            # and still add the raw quantity so nothing is silently dropped.
            entry["required_qty"] += req["required_qty"]

    totals = {shop: 0.0 for shop in shops}
    missing = {shop: 0 for shop in shops}
    missing_items = {shop: [] for shop in shops}
    items = []

    for key, agg in aggregated.items():
        shop_options_for_key = options.get(key, {})

        item_missing = {}
        chosen = {}

        for shop in shops:
            shop_opts = shop_options_for_key.get(shop, [])
            best = best_option_for_shop(shop_opts, agg["required_qty"], agg["unit_type"])

            if best is None:
                missing[shop] += 1
                missing_items[shop].append(agg["display_name"])
                item_missing[shop] = True
            else:
                totals[shop] += best["total_price"]
                chosen[shop] = best
                item_missing[shop] = False

        items.append(
            {
                "display_name": agg["display_name"],
                "required_qty": agg["required_qty"],
                "unit_type": agg["unit_type"],
                "missing": item_missing,
                "chosen": chosen,
            }
        )

    items.sort(key=lambda x: str(x["display_name"]).lower())

    best_shop = sorted(shops, key=lambda s: (missing[s], totals[s]))[0]

    return {
        "totals": totals,
        "missing": missing,
        "missing_items": missing_items,
        "ingredient_price_details": items,
        "items": items,
        "best_shop": best_shop,
        "best_total": totals[best_shop],
        "best_missing": missing[best_shop],
        "best_missing_items": missing_items[best_shop],
        "has_prices": True,
    }


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
        return f"{summary['best_shop']} {format_price(summary['best_total'])}"

    return f"{summary['best_shop']} {format_price(summary['best_total'])} ({missing} N/A)"


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


# ══════════════════════════════════════════════════════════════════════════════
# MEAL PLAN GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def build_multiple_plans(
    df: pd.DataFrame,
    n_plans: int,
    n_meals: int,
    n_desserts: int,
    n_drinks: int,
    n_sides: int,
    excluded_types: list[str],
    excluded_recipes: list[str],
    max_pasta_per_plan: int,
    sim_matrix: np.ndarray,
    sort_by: str,
    candidate_pool_size: int,
    recipe_price_summaries: dict | None = None,
) -> list[tuple[list[int], float]]:
    """
    Build meal plans from a smaller candidate pool.

    If sorting by cheapest:
    - price each individual recipe first
    - keep the cheapest candidate recipes
    - build combinations from those
    - sort plans by exact total price

    If sorting by shared ingredient score:
    - score each recipe by overlap potential
    - keep the highest-overlap candidate recipes
    - build combinations from those
    - sort plans by exact shared ingredient score
    """

    recipe_price_summaries = recipe_price_summaries or {}

    type_col = df["Type"].str.lower()

    dessert_mask = type_col.str.contains(
        "dessert|cake|bake|sweet|pudding|cookie|brownie",
        na=False,
    )

    drink_mask = type_col.str.contains(
        "drink|smoothie|juice|shake",
        na=False,
    )

    side_mask = type_col.str.contains(
        "side|light|snack|starter",
        na=False,
    ) & ~dessert_mask & ~drink_mask

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
    side_idx = df.index[side_mask & ~excluded_mask].tolist()
    main_idx = df.index[~dessert_mask & ~drink_mask & ~side_mask & ~excluded_mask].tolist()

    n_main = int(n_meals) - int(n_desserts) - int(n_drinks) - int(n_sides)

    if n_main < 0:
        return []

    if len(main_idx) < n_main:
        return []

    if len(dessert_idx) < int(n_desserts):
        return []

    if len(drink_idx) < int(n_drinks):
        return []

    if len(side_idx) < int(n_sides):
        return []

    def pasta_count(indices):
        return sum(is_pasta_recipe(df.loc[i, "Name"]) for i in indices)

    def recipe_price_key(index):
        summary = recipe_price_summaries.get(index)

        if not summary or not summary.get("has_prices"):
            return (999999, 999999.0)

        return (
            int(summary.get("best_missing", 999999)),
            float(summary.get("best_total", 999999.0)),
        )

    def recipe_overlap_score(index):
        tokens = ingredient_tokens(df.loc[index, "Ingredients"])

        if not tokens:
            return 0

        # Score recipes higher if their ingredients appear in many other recipes.
        eligible_indices = main_idx + dessert_idx + drink_idx + side_idx
        all_tokens = []

        for i in eligible_indices:
            all_tokens.extend(ingredient_tokens(df.loc[i, "Ingredients"]))

        token_counts = pd.Series(all_tokens).value_counts().to_dict()

        return sum(token_counts.get(token, 0) for token in tokens)

    def choose_candidates(indices):
        if not indices:
            return []

        if sort_by == "Cheapest meal plan":
            ordered = sorted(indices, key=recipe_price_key)
        else:
            ordered = sorted(indices, key=recipe_overlap_score, reverse=True)

        return ordered[:int(candidate_pool_size)]

    main_idx = choose_candidates(main_idx)
    dessert_idx = choose_candidates(dessert_idx)
    drink_idx = choose_candidates(drink_idx)
    side_idx = choose_candidates(side_idx)

    if len(main_idx) < n_main:
        return []

    if len(dessert_idx) < int(n_desserts):
        return []

    if len(drink_idx) < int(n_drinks):
        return []

    if len(side_idx) < int(n_sides):
        return []

    main_combos = list(combinations(main_idx, n_main))

    dessert_combos = [()] if int(n_desserts) == 0 else list(combinations(dessert_idx, int(n_desserts)))
    drink_combos = [()] if int(n_drinks) == 0 else list(combinations(drink_idx, int(n_drinks)))
    side_combos = [()] if int(n_sides) == 0 else list(combinations(side_idx, int(n_sides)))

    scored_plans = []

    for main_combo in main_combos:
        for dessert_combo in dessert_combos:
            for drink_combo in drink_combos:
                for side_combo in side_combos:
                    selected = (
                        list(main_combo)
                        + list(dessert_combo)
                        + list(drink_combo)
                        + list(side_combo)
                    )

                    if pasta_count(selected) > int(max_pasta_per_plan):
                        continue

                    plan_recipes = df.iloc[selected].reset_index(drop=True)
                    coverage_score, _ = plan_ingredient_coverage_score(plan_recipes)

                    if sort_by == "Cheapest meal plan":
                        plan_missing = 0
                        plan_total = 0.0

                        for idx in selected:
                            summary = recipe_price_summaries.get(idx)

                            if summary and summary.get("has_prices"):
                                plan_missing += int(summary.get("best_missing", 999999))
                                plan_total += float(summary.get("best_total", 999999.0))
                            else:
                                plan_missing += 999999
                                plan_total += 999999.0

                        sort_key = (plan_missing, plan_total)
                    else:
                        sort_key = (-coverage_score,)

                    scored_plans.append(
                        {
                            "indices": selected,
                            "coverage_score": coverage_score,
                            "sort_key": sort_key,
                        }
                    )

    scored_plans.sort(key=lambda x: x["sort_key"])
    scored_plans = scored_plans[:int(n_plans)]

    return [
        (plan["indices"], plan["coverage_score"])
        for plan in scored_plans
    ]


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


def generate_pdf(
    plan_recipes: pd.DataFrame,
    serving_overrides: dict,
    plan_num: int,
    recipe_price_summaries: dict | None = None,
    plan_shopping: dict | None = None,
) -> bytes:
    """Generate a PDF of the meal plan."""

    recipe_price_summaries = recipe_price_summaries or {}

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

    if plan_shopping and plan_shopping.get("has_prices") and plan_shopping.get("best_shop"):
        pdf.cell(
            0, 6,
            pdf_line(f"Best total price: {format_shop_price(plan_shopping)}"),
            ln=True, align="C",
        )

    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)

    # ── Full shopping list (aggregated across the whole plan, whole packages) ──
    best_shop = plan_shopping.get("best_shop") if plan_shopping else None
    shopping_items = plan_shopping.get("items", []) if plan_shopping else []

    if shopping_items and best_shop:
        pdf.set_x(pdf.l_margin)
        pdf.set_fill_color(44, 44, 44)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 9, pdf_line(f"Full Shopping List - {best_shop}"), fill=True)
        pdf.set_text_color(0, 0, 0)

        pdf.set_font("Helvetica", "", 9)

        for item in shopping_items:
            name = item["display_name"] or ""
            need_text = format_quantity(item["required_qty"], item["unit_type"])
            chosen = item.get("chosen", {}).get(best_shop)
            is_missing = item.get("missing", {}).get(best_shop, True)

            if is_missing or not chosen:
                line = f"- {name} (need {need_text}): N/A"
            else:
                size_text = format_quantity(chosen["size_qty"], chosen["unit_type"])
                line = (
                    f"- {name} (need {need_text}): "
                    f"{chosen['packs']} x {chosen['name']} ({size_text}) "
                    f"@ {format_price(chosen['price'])} each = {format_price(chosen['total_price'])}"
                )

            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5, pdf_line(line))

        pdf.ln(2)

        list_total = plan_shopping["totals"].get(best_shop, 0.0)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(
            0, 6,
            pdf_line(f"Shopping list total ({best_shop}): {format_price(list_total)}"),
        )

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

        recipe_price_summary = recipe_price_summaries.get(row["Name"])
        recipe_price_text = (
            format_shop_price(recipe_price_summary)
            if recipe_price_summary else "No price data"
        )

        meta = (
            f"{row['Type']} | Serves {desired} | {tried_label} | "
            f"Best price (this recipe alone): {recipe_price_text}"
        )

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
        missing_items = price_summary.get("best_missing_items", [])

        if missing_items:
            missing_text = "Missing prices: " + ", ".join(missing_items)
        else:
            missing_text = ""
    else:
        price_text = "No price data"
        missing_text = ""

    missing_html = ""

    if missing_text:
        missing_html = f"""
      <div style="margin-top:2px;font-size:10.5px;color:{COLOURS['muted']};font-style:italic;">
        {html.escape(missing_text)}
      </div>
        """

    st.markdown(
        f"""
    <div class="recipe-card {tried_class}">
      <div style="font-family:'DM Serif Display',serif;font-size:15px;font-weight:600;color:{COLOURS['text']}">
        {html.escape(str(row['Name']))}
      </div>
      <div style="margin-top:4px;">
        <span class="badge {tried_badge}">{html.escape(str(row['Tried']))}</span>
        <span class="type-chip">{html.escape(str(row['Type']))}</span>
      </div>
      <div style="margin-top:8px;font-size:12px;color:{COLOURS['muted']}">
        Serves <b style="color:{COLOURS['text']}">{serving_override}</b>
        &nbsp;·&nbsp;
        {n_ing} ingredients
      </div>
      <div class="price-line">
        Best price (this recipe alone): <b>{html.escape(price_text)}</b>
      </div>
      {missing_html}
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.expander("Open recipe"):
        caption_text = (
            f"Serves: {serving_override}  |  "
            f"Type: {row['Type']}  |  "
            f"Status: {row['Tried']}  |  "
            f"Best price: {price_text}"
        )

        if missing_text:
            caption_text += f"  |  {missing_text}"

        st.caption(caption_text)

        st.markdown("---")

        col_i, col_s = st.columns([1, 1])

        with col_i:
            st.markdown("**Ingredients**")

            ingredient_price_details = []

            if price_summary and price_summary.get("has_prices"):
                ingredient_price_details = price_summary.get("ingredient_price_details", [])

            best_shop = None

            if price_summary and price_summary.get("has_prices"):
                best_shop = price_summary.get("best_shop")

            for i, ing in enumerate(scaled):
                price_note = ""

                if best_shop and i < len(ingredient_price_details):
                    detail = ingredient_price_details[i]
                    is_missing = detail.get("missing", {}).get(best_shop, True)
                    chosen = detail.get("chosen", {}).get(best_shop)

                    if is_missing or not chosen:
                        price_note = f"{best_shop}: N/A"
                    else:
                        size_text = format_quantity(chosen["size_qty"], chosen["unit_type"])
                        price_note = (
                            f"{best_shop}: {format_price(chosen['total_price'])} "
                            f"({chosen['packs']} x {chosen['name']}, {size_text})"
                        )

                if price_note:
                    st.markdown(
                        f"- {html.escape(str(ing))} "
                        f"<span style='font-size:11px;color:{COLOURS['muted']};'>"
                        f"{html.escape(price_note)}"
                        f"</span>",
                        unsafe_allow_html=True,
                    )
                else:
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
        price_options = load_prices()
    except Exception as e:
        price_options = {"shops": [], "options": {}}
        st.warning("Recipes loaded, but the Price 2 sheet could not be loaded.")
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
        "Number of meal plans to show",
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

    n_sides = st.sidebar.number_input(
        "Light/side dishes per plan",
        min_value=0,
        max_value=5,
        value=0,
        help="Recipes whose Type contains 'side', 'light', 'snack', or 'starter'.",
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

    candidate_pool_size = st.sidebar.number_input(
        "Candidate recipes to check",
        min_value=5,
        max_value=len(df),
        value=min(50, len(df)),
        help=(
            "For cheapest plans, this keeps the cheapest individual recipes. "
            "For shared ingredient score, this keeps the recipes with the highest ingredient-overlap potential."
        ),
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

    default_side_serving = st.sidebar.number_input(
        "Default servings for sides",
        min_value=1,
        max_value=20,
        value=2,
    )

    if "serving_overrides" not in st.session_state:
        st.session_state["serving_overrides"] = {}

    def recipe_category(recipe_type):
        recipe_type = str(recipe_type).lower()

        if re.search(r"dessert|cake|bake|sweet|pudding|cookie|brownie", recipe_type):
            return "Dessert"

        if re.search(r"drink|smoothie|juice|shake", recipe_type):
            return "Drink"

        if re.search(r"side|light|snack|starter", recipe_type):
            return "Side"

        return "Meal"

    def default_serving_for_type(recipe_type):
        category = recipe_category(recipe_type)

        if category == "Drink":
            return default_drink_serving

        if category == "Side":
            return default_side_serving

        return default_main_serving

    meal_rows = df[df["Type"].apply(recipe_category) == "Meal"]
    dessert_rows = df[df["Type"].apply(recipe_category) == "Dessert"]
    drink_rows = df[df["Type"].apply(recipe_category) == "Drink"]
    side_rows = df[df["Type"].apply(recipe_category) == "Side"]

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

    with st.sidebar.expander("Side serving overrides"):
        if side_rows.empty:
            st.caption("No side/light recipes found.")
        else:
            for _, row in side_rows.iterrows():
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

    n_priced_ingredients = len(price_options.get("options", {}))
    n_shops = len(price_options.get("shops", []))

    if n_shops == 0:
        st.caption(f"{len(df)} recipes loaded from Google Sheets. No price data loaded yet.")
    else:
        st.caption(
            f"{len(df)} recipes loaded from Google Sheets. "
            f"{n_priced_ingredients} priced ingredients across {n_shops} shops loaded from Price 2 sheet."
        )

    if st.button("Generate Meal Plans", use_container_width=True):
        with st.spinner("Analysing ingredients and building meal plans..."):
            sim = compute_similarity_matrix(tuple(df["Ingredients"].tolist()))

            recipe_price_summaries_for_generation = {}

            if sort_by == "Cheapest meal plan":
                for idx, row in df.iterrows():
                    serving = get_serving(row["Name"])
                    recipe_price_summaries_for_generation[idx] = calculate_recipe_price(
                        row,
                        serving,
                        price_options,
                    )

            plans = build_multiple_plans(
                df=df,
                n_plans=int(n_plans),
                n_meals=int(n_meals),
                n_desserts=int(n_desserts),
                n_drinks=int(n_drinks),
                n_sides=int(n_sides),
                excluded_types=excluded_types,
                excluded_recipes=excluded_recipes,
                max_pasta_per_plan=int(max_pasta_per_plan),
                sim_matrix=sim,
                sort_by=sort_by,
                candidate_pool_size=int(candidate_pool_size),
                recipe_price_summaries=recipe_price_summaries_for_generation,
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
    if not plans:
        st.warning("No valid meal plans were found with the current filters.")
        return

    # Build display data, including fresh scores and prices.
    plan_display = []

    for recipe_indices, score in plans:
        plan_recipes = df.iloc[recipe_indices].reset_index(drop=True)
        coverage_score, coverage_details = plan_ingredient_coverage_score(plan_recipes)

        # Per-recipe isolated pricing, for each card's "best price" line.
        recipe_price_summaries = {}

        for _, row in plan_recipes.iterrows():
            serving = get_serving(row["Name"])
            recipe_price_summaries[row["Name"]] = calculate_recipe_price(
                row,
                serving,
                price_options,
            )

        # Plan-level pricing: aggregate raw quantities across all recipes
        # BEFORE rounding to whole packages, so shared ingredients aren't
        # bought multiple times.
        plan_shopping = build_plan_shopping_list(
            plan_recipes,
            get_serving,
            price_options,
        )

        plan_display.append(
            {
                "recipe_indices": recipe_indices,
                "plan_recipes": plan_recipes,
                "coverage_score": coverage_score,
                "coverage_details": coverage_details,
                "recipe_price_summaries": recipe_price_summaries,
                "plan_shopping": plan_shopping,
            }
        )

    st.caption(
        f"Showing top {len(plan_display)} meal plans using a candidate pool of up to "
        f"{int(candidate_pool_size)} recipes per recipe group."
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
        plan_shopping = plan["plan_shopping"]

        pct = int(coverage_score * 100)
        best_price_text = format_shop_price(plan_shopping)

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
            format_all_shop_prices(plan_shopping),
            unsafe_allow_html=True,
        )

        with st.expander("See full shopping list (whole packages, best shop)"):
            best_shop = plan_shopping.get("best_shop")
            shopping_items = plan_shopping.get("items", [])

            if not best_shop or not shopping_items:
                st.write("No price data loaded.")
            else:
                for item in shopping_items:
                    need_text = format_quantity(item["required_qty"], item["unit_type"])
                    chosen = item.get("chosen", {}).get(best_shop)
                    is_missing = item.get("missing", {}).get(best_shop, True)

                    if is_missing or not chosen:
                        st.markdown(f"- **{item['display_name']}** (need {need_text}): N/A")
                    else:
                        size_text = format_quantity(chosen["size_qty"], chosen["unit_type"])
                        st.markdown(
                            f"- **{item['display_name']}** (need {need_text}): "
                            f"{chosen['packs']} x {chosen['name']} ({size_text}) "
                            f"@ {format_price(chosen['price'])} each = "
                            f"**{format_price(chosen['total_price'])}**"
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
            recipe_price_summaries=recipe_price_summaries,
            plan_shopping=plan_shopping,
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
            search_terms = [
                term.strip()
                for term in search.split(",")
                if term.strip()
            ]

            for term in search_terms:
                mask &= (
                    df["Name"].str.contains(term, case=False, na=False, regex=False)
                    | df["Ingredients"].str.contains(term, case=False, na=False, regex=False)
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
            recipe_price_summary = calculate_recipe_price(row, serving, price_options)

            with cols[i % 3]:
                render_recipe_card(
                    row,
                    serving,
                    f"browse_{i}",
                    recipe_price_summary,
                )


if __name__ == "__main__":
    main()
