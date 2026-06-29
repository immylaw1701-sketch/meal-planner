import streamlit as st
import pandas as pd
import numpy as np
import re
import textwrap
import gspread

from google.oauth2.service_account import Credentials
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


# ── Google Sheet settings ───────────────────────────────────────────────────
SPREADSHEET_NAME = "Meal plan"
WORKSHEET_NAME = "Sheet1"


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

    col_map = {}

    for c in df.columns:
        lc = str(c).lower()

        if "name" in lc:
            col_map[c] = "Name"
        elif "serving" in lc:
            col_map[c] = "Servings"
        elif "ingredient" in lc:
            col_map[c] = "Ingredients"
        elif "step" in lc or "method" in lc:
            col_map[c] = "Steps"
        elif "tried" in lc:
            col_map[c] = "Tried"
        elif "type" in lc:
            col_map[c] = "Type"

    df = df.rename(columns=col_map)

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


@st.cache_data(ttl=300)
def load_recipes() -> pd.DataFrame:
    """Load recipe data directly from Google Sheets."""

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]

    service_account_info = dict(st.secrets["gcp_service_account"])

    creds = Credentials.from_service_account_info(
        service_account_info,
        scopes=scopes,
    )

    gc = gspread.authorize(creds)

    spreadsheet = gc.open(SPREADSHEET_NAME)
    worksheet = spreadsheet.worksheet(WORKSHEET_NAME)

    data = worksheet.get_all_records()
    raw_df = pd.DataFrame(data)

    return clean_df(raw_df)


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
# SIMILARITY AND MEAL PLAN GENERATION
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
        "and", "or", "a", "an", "the",
    }

    words = [w for w in token.split() if w not in stop_words and len(w) > 2]

    replacements = {
        "chickens": "chicken",
        "breasts": "chicken",
        "thighs": "chicken",
        "onions": "onion",
        "tomatoes": "tomato",
        "potatoes": "potato",
        "peppers": "pepper",
        "noodles": "noodle",
        "eggs": "egg",
        "cloves": "garlic",
        "garlics": "garlic",
    }

    words = [replacements.get(w, w) for w in words]

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


def generate_meal_plan(
    df: pd.DataFrame,
    n_meals: int,
    n_desserts: int,
    n_drinks: int,
    excluded_types: list[str],
    sim_matrix: np.ndarray,
) -> tuple[list[int], float]:
    """Generate one meal plan."""

    type_col = df["Type"].str.lower()

    dessert_mask = type_col.str.contains("dessert|cake|bake|sweet|pudding|cookie|brownie", na=False)
    drink_mask = type_col.str.contains("drink|smoothie|juice|shake", na=False)

    if excluded_types:
        excluded_lower = [e.lower() for e in excluded_types]
        excluded_mask = type_col.isin(excluded_lower)
    else:
        excluded_mask = pd.Series([False] * len(df), index=df.index)

    dessert_idx = df.index[dessert_mask & ~excluded_mask].tolist()
    drink_idx = df.index[drink_mask & ~excluded_mask].tolist()
    main_idx = df.index[~dessert_mask & ~drink_mask & ~excluded_mask].tolist()

    selected = []

    selected.extend(dessert_idx[:n_desserts])
    selected.extend(drink_idx[:n_drinks])

    n_main = n_meals - len(selected)

    if n_main < 0:
        n_main = 0

    if len(main_idx) >= 1 and n_main >= 1:
        if n_main == 1:
            selected.append(main_idx[0])
        else:
            best_pair = (
                main_idx[0],
                main_idx[1] if len(main_idx) > 1 else main_idx[0],
            )
            best_sim = -1

            for a, b in combinations(main_idx[:30], 2):
                s = sim_matrix[a, b]

                if s > best_sim:
                    best_sim = s
                    best_pair = (a, b)

            chosen_main = list(best_pair)
            remaining = [i for i in main_idx if i not in chosen_main]

            while len(chosen_main) < n_main and remaining:
                avg_sims = [
                    np.mean([sim_matrix[r, c] for c in chosen_main])
                    for r in remaining
                ]

                best_next = remaining[int(np.argmax(avg_sims))]
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

def render_recipe_card(row: pd.Series, serving_override: int, key_prefix: str):
    """Render a recipe box using a Streamlit expander."""

    tried_value = str(row["Tried"]).lower()

    tried_class = "tried" if tried_value == "tried" else "not-tried"
    tried_badge = "badge-tried" if tried_value == "tried" else "badge-nottried"

    scaled = scale_ingredients(row["Ingredients"], row["Servings"], serving_override)
    steps = [s.strip() for s in str(row["Steps"]).split(";") if s.strip()]
    n_ing = len(scaled)

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
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.expander("Open recipe"):
        st.caption(
            f"Serves: {serving_override}  |  "
            f"Type: {row['Type']}  |  "
            f"Status: {row['Tried']}"
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
        st.error("Could not load recipes from Google Sheets.")
        st.exception(e)
        return

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

    st.caption(f"{len(df)} recipes loaded from Google Sheets.")

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

    st.markdown(
        f"""
    <div style='display:flex;gap:20px;align-items:center;margin-bottom:12px;'>
      <span><span class='legend-dot' style='background:{COLOURS["Tried"]}'></span>Tried</span>
      <span><span class='legend-dot' style='background:{COLOURS["Not Tried"]}'></span>Not Tried</span>
      <span style='color:{COLOURS["muted"]};font-size:12px;'>
        Plans are ordered by shared ingredient score
      </span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    for plan_idx, (recipe_indices, score) in enumerate(plans):
        plan_recipes = df.iloc[recipe_indices].reset_index(drop=True)

        coverage_score, coverage_details = plan_ingredient_coverage_score(plan_recipes)
        pct = int(coverage_score * 100)

        st.markdown(
            f"""
        <div class='plan-row-header'>
          Plan {plan_idx + 1}
          <span style='font-weight:300;font-size:11px;margin-left:12px;opacity:0.7;'>
            Shared ingredient score
          </span>
          <span style='font-weight:700;font-size:12px;margin-left:4px;'>{pct}%</span>
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

            with cols[col_idx % n_cols]:
                render_recipe_card(
                    row,
                    serving,
                    f"plan{plan_idx}_rec{col_idx}",
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

            with cols[i % 3]:
                render_recipe_card(row, serving, f"browse_{i}")


if __name__ == "__main__":
    main()