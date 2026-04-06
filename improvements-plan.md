# Used Car Price Prediction — Repository Improvements Plan

## Context

This is a portfolio ML project predicting used car prices using a CatBoost model trained on 445 records scraped from CarDekho. The repo has solid ML fundamentals (proper train/test split, multiple model comparison, log-target transformation) but suffers from critical deployment issues, missing ML rigor, and code quality problems that undermine its portfolio value.

**Goal:** Make the repo production-quality and portfolio-ready by fixing broken functionality, strengthening ML practices, improving code quality, and cleaning up project organization.

---

## Priority 1 — Critical Fixes (Broken Deployment)

### 1.1 Fix `requirements.txt` Encoding Corruption
**File:** `/home/user/Used-car-price-predicition-model/requirements.txt`

The file is UTF-16 encoded with spaces between every character. Replace entirely with a clean file:
```
catboost==1.2.8
scikit-learn>=1.3.0
category-encoders>=2.6.0
pandas>=2.0.0
numpy>=1.26.0
streamlit>=1.30.0

# Dev/notebook dependencies
# matplotlib seaborn plotly scipy statsmodels selenium beautifulsoup4
```

### 1.2 Fix `.gitignore` to Track the Model File
**File:** `/home/user/Used-car-price-predicition-model/.gitignore`

The trained model is in `.gitignore` under `models/`, which means anyone cloning the repo gets a broken app. Fix by:
- Change `models/` to `catboost_info/` (the training log directory, not the model itself)
- Add `models/catboost_model.cbm` as a tracked file OR add a `scripts/train.py` so users can retrain

### 1.3 Add Model Saving to main.ipynb
**File:** `/home/user/Used-car-price-predicition-model/main.ipynb`

Cell 81 calls `catb_model.save_model("models/catboost_model.cbm")` but the `models/` directory doesn't exist — `os.makedirs("models", exist_ok=True)` must precede the save call. Also, the notebook trains `catb_model` twice (cell 54: raw price, cell 59: log price) — add a comment clarifying only the log-price model is saved.

### 1.4 Fix app.py Brand List
**File:** `/home/user/Used-car-price-predicition-model/app.py`

The dropdown has 14 brands; the dataset has 22. Missing: Citroen, Datsun, Jaguar, Jeep, Land Rover, MG, Mercedes-Benz, Volvo. These missing brands will silently produce wrong predictions.

**Fix:** Derive the brand list from the cleaned CSV at startup:
```python
brands = sorted(pd.read_csv("data/cleaned_cardekho_used_cars.csv")["brand"].unique().tolist())
```

### 1.5 Fix Misleading Confidence Interval in app.py
**File:** `/home/user/Used-car-price-predicition-model/app.py`

The current ±10% band is labeled as a "confidence range" but only 50.6% of test predictions fall within ±10%. This is actively misleading.

**Fix:** Change the label to "Estimated Range" and use ±₹1.5 lakh (derived from the actual test set residual std dev of ₹213K, roughly 1.5 standard deviation bounds):
```python
lower = max(0, prediction - 150000)
upper = prediction + 150000
st.write(f"Estimated Range: ₹{lower/100000:.2f}L – ₹{upper/100000:.2f}L")
st.caption("Note: Model predictions within ±₹1.5L ~73% of the time on test data")
```

---

## Priority 2 — ML Quality Improvements

### 2.1 Add K-Fold Cross-Validation
**File:** `/home/user/Used-car-price-predicition-model/main.ipynb`

With only 445 samples, a single 80/20 split is high variance. Add 5-fold CV after the existing split:
```python
from sklearn.model_selection import cross_val_score, KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='r2')
print(f"CV R² scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

### 2.2 Complete the Hyperparameter Tuning (Dead Import Fix)
**File:** `/home/user/Used-car-price-predicition-model/main.ipynb`

`GridSearchCV` is imported in cell 0 but never used. Either remove the import or implement a basic hyperparameter search:
```python
from sklearn.model_selection import RandomizedSearchCV
param_grid = {
    'iterations': [500, 1000, 1500],
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
}
# Use RandomizedSearchCV to avoid full grid search on small dataset
```

### 2.3 Add Outlier Detection/Treatment
**File:** `/home/user/Used-car-price-predicition-model/main.ipynb`

Currently only `dropna()` is used. Add IQR-based outlier flagging for `km_driven` and `price`:
```python
Q1, Q3 = df['km_driven'].quantile([0.25, 0.75])
IQR = Q3 - Q1
outliers = df[(df['km_driven'] < Q1 - 1.5*IQR) | (df['km_driven'] > Q3 + 1.5*IQR)]
print(f"Outliers in km_driven: {len(outliers)}")
# Decide: remove or cap (Winsorization)
```

---

## Priority 3 — Code Quality

### 3.1 Remove the Double-Training Confusion
**File:** `/home/user/Used-car-price-predicition-model/main.ipynb`

Cells 54 and 59 both call `catb_model.fit()`. The first fit is on raw price (experimental), the second on log-transformed price (final). Add a markdown cell between them explaining this is intentional A/B comparison, and that only the log-transformed model is retained.

### 3.2 Fix Inconsistent Formatting
**File:** `/home/user/Used-car-price-predicition-model/main.ipynb`

- Cell with `df_copy.drop(..., inplace=  True)` — double space
- Variables named `x`, `y`, `s` in plotting cells
- Add comments to code cells that have 5+ lines with no explanation

### 3.3 Remove Debug File Writing in scrapper.ipynb
**File:** `/home/user/Used-car-price-predicition-model/scrapper.ipynb`

Remove the line writing `debug_page.html` to disk on every page scrape.

### 3.4 Add headless mode to scrapper.ipynb
**File:** `/home/user/Used-car-price-predicition-model/scrapper.ipynb`

Add `options.add_argument("--headless")` so scraper works in server environments.

---

## Priority 4 — Project Organization

### 4.1 Create `notebooks/` Directory
Move notebooks from root to organized subdirectory:
```
notebooks/
├── 01_eda.ipynb        (rename 02_eda.ipynb)
├── 02_main.ipynb       (rename main.ipynb)
└── 03_scrapper.ipynb   (rename scrapper.ipynb)
```

Update README references accordingly.

### 4.2 Update README.md
**File:** `/home/user/Used-car-price-predicition-model/README.md`
- Fix project structure diagram to show correct file locations
- Remove claims that model is "saved" (it isn't committed)
- Add instructions to run `python train.py` or re-run notebook to generate model
- Fix performance metrics to include CV scores (not just single-split)
- Add note about the 22-brand coverage

### 4.3 Add `models/` Directory Placeholder
Create `models/.gitkeep` and update `.gitignore` to track the model file or provide clear instructions for generating it.

---

## Implementation Order

1. **Fix requirements.txt** (5 min) — unblocks pip install
2. **Fix .gitignore + create models/ dir** (5 min) — unblocks deployment
3. **Fix app.py brand list** (5 min) — fixes silent prediction errors
4. **Fix confidence interval in app.py** (10 min) — fixes misleading UX
5. **Add os.makedirs before model save in main.ipynb** (5 min) — fixes model save
6. **Add cross-validation cells to main.ipynb** (20 min) — ML rigor
7. **Remove dead GridSearchCV import or implement tuning** (15 min)
8. **Add outlier detection cell** (15 min)
9. **Fix formatting/comments in main.ipynb** (20 min)
10. **Fix scrapper.ipynb debug file + headless** (10 min)
11. **Update README.md** (15 min)
12. **Move notebooks to notebooks/ dir** (5 min)

---

## Verification

- `pip install -r requirements.txt` should succeed cleanly
- `streamlit run app.py` should launch without ImportError or FileNotFoundError
- App dropdown should show all 22 brands from the dataset
- Confidence interval label should say "Estimated Range" not "Confidence Range"
- main.ipynb cells should run top-to-bottom without errors
- models/ directory should exist and contain catboost_model.cbm after running notebook
- CV R² score should be reported alongside the test R² of 0.793

---

## Critical Files to Modify

| File | Changes |
|------|---------|
| `requirements.txt` | Full rewrite — fix encoding corruption |
| `.gitignore` | Change `models/` to `catboost_info/` |
| `app.py` | Fix brand list (line ~30), fix confidence interval (line ~85) |
| `main.ipynb` | Add `os.makedirs`, add CV cells, fix double-training comment, fix formatting |
| `scrapper.ipynb` | Remove debug file write, add headless flag |
| `README.md` | Fix project structure, add CV metrics, fix model file instructions |
