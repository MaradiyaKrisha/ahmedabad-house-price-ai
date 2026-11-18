import re
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

MONTH_TOKENS = {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"}


def clean_location(val) -> str:
    if pd.isna(val):
        return "Other"
    text = str(val).strip()
    if not text or text.lower() in {"nan", "none", "null", "na", "n/a", "-"}:
        return "Other"

    # Remove scraped "in " / "in" prefix (e.g., "inshilaj, ahmedabad" -> "shilaj, ahmedabad")
    text = re.sub(r"^\s*in(?=[a-zA-Z])", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*in\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" ,.-")

    lower = text.lower()
    # Filter out pure numbers, BHK-like fragments ("4, 5"), or month names ("dec", "jan")
    if not text or lower in MONTH_TOKENS or re.fullmatch(r"[\d\s,./-]+", text):
        return "Other"

    return text.title()


def clean_category(val, default: str) -> str:
    if pd.isna(val):
        return default
    text = re.sub(r"\s+", " ", str(val).strip())
    if not text or text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return default
    return text.title()


# 1. Load Real Ahmedabad Dataset
df = pd.read_csv("housing_data.csv")

# 2. Select relevant columns & clean data
cols = ["location", "area_in_sqft", "area_type", "bhk_type", "property_type", "price_in_cr"]
df = df[cols].copy()

df["area_in_sqft"] = pd.to_numeric(df["area_in_sqft"], errors="coerce")
df["bhk_type"] = pd.to_numeric(df["bhk_type"], errors="coerce")
df["price_in_cr"] = pd.to_numeric(df["price_in_cr"], errors="coerce")

df["location"] = df["location"].apply(clean_location)
df["property_type"] = df["property_type"].apply(lambda x: clean_category(x, "Flat"))
df["area_type"] = df["area_type"].apply(lambda x: clean_category(x, "Super"))

# Preserve Land/Plot rows by assigning bhk_type = 0 instead of dropping them
is_land = df["property_type"].str.lower().str.contains("land|plot", na=False)
df.loc[is_land, "bhk_type"] = 0

df = df.dropna(subset=["price_in_cr", "area_in_sqft", "bhk_type"])
df["bhk_type"] = df["bhk_type"].astype(int)

# 3. Domain-Specific Outlier Removal (Price per Sqft IQR)
df = df[(df["area_in_sqft"] >= 250) & (df["area_in_sqft"] <= 15000)]
df = df[(df["price_in_cr"] >= 0.08) & (df["price_in_cr"] <= 15)]
df["calc_pps"] = (df["price_in_cr"] * 1e7) / df["area_in_sqft"]
q1, q3 = df["calc_pps"].quantile([0.03, 0.97])
df = df[(df["calc_pps"] >= q1) & (df["calc_pps"] <= q3)].drop(columns=["calc_pps"])

# Group rare locations (< 8 listings) into 'Other'
loc_counts = df["location"].value_counts()
rare_locs = loc_counts[loc_counts < 8].index
df["location"] = df["location"].apply(lambda x: "Other" if x in rare_locs else x)

X = df[["location", "area_in_sqft", "area_type", "bhk_type", "property_type"]]
y = df["price_in_cr"]

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Preprocessing + Tuned XGBoost Pipeline
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), ["area_in_sqft", "bhk_type"]),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), ["location", "area_type", "property_type"])
])

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(
        n_estimators=350,
        learning_rate=0.04,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42
    ))
])

# 6. Train & Evaluate
model_pipeline.fit(X_train, y_train)
preds = model_pipeline.predict(X_test)
r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)
print(f"Upgraded R2 Score: {r2:.4f}")
print(f"MAE: INR {mae:.2f} Cr")
print(f"Cleaned Locations: {df['location'].nunique()} | Samples: {len(df)}")

# 7. Save Model & Metadata
locations_list = sorted([loc for loc in df["location"].unique() if loc != "Other"]) + ["Other"]
prop_types_list = sorted(df["property_type"].unique().tolist())
area_types_list = sorted(df["area_type"].unique().tolist())

joblib.dump({
    "model": model_pipeline,
    "locations": locations_list,
    "property_types": prop_types_list,
    "area_types": area_types_list,
    "metrics": {"r2": round(r2 * 100, 1), "mae": round(mae, 2), "samples": len(df)}
}, "house_price_pipeline.pkl")

print("High-Accuracy Ahmedabad Model saved successfully!")