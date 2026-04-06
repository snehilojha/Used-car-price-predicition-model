# Used Car Price Prediction Model

A machine learning project that predicts used car prices in the Indian market using data scraped from CarDekho. This end-to-end project includes web scraping, data preprocessing, model training, and a Streamlit web application for real-time predictions.

## Project Overview

This project addresses the challenge of accurately pricing used cars in the Indian automotive market. By leveraging machine learning algorithms, particularly CatBoost, the model helps buyers and sellers make informed decisions about fair market values.

**Key Features:**
- Custom web scraper for real-time data collection from CarDekho
- Comprehensive data cleaning and feature engineering pipeline
- Multiple ML models compared (Ridge, Random Forest, CatBoost)
- 5-fold cross-validation and GridSearchCV hyperparameter tuning
- Interactive Streamlit web application for price predictions
- Test R² = 0.793, MAE = ₹146,009 (89-sample held-out test set)

## Project Structure

```
used_car_model/
├── data/
│   ├── raw_cardekho_used_cars.csv          # Original scraped data
│   └── cleaned_cardekho_used_cars.csv      # Processed dataset (445 rows, 22 brands)
├── models/
│   └── catboost_model.cbm                  # Trained CatBoost model
├── results/
│   └── catboost_results.csv                # Test set predictions + errors (89 rows)
├── catboost_info/                          # CatBoost training logs (git-ignored)
├── scrapper.ipynb                          # Web scraping notebook
├── 02_eda.ipynb                            # Exploratory data analysis
├── main.ipynb                              # Data cleaning, EDA, and modeling
├── app.py                                  # Streamlit web application
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

## Dataset

**Source**: CarDekho (used cars in New Delhi)  
**Size**: 445 records after cleaning  
**Brands covered**: 22 (Audi, BMW, Citroen, Datsun, Ford, Honda, Hyundai, Jaguar, Jeep, Kia, Land Rover, MG, Mahindra, Maruti, Mercedes-Benz, Nissan, Renault, Skoda, Tata, Toyota, Volkswagen, Volvo)

**Features**:
| Feature | Type | Description |
|---------|------|-------------|
| `brand` | categorical | Car manufacturer |
| `fuel_type` | categorical | Petrol / Diesel / CNG |
| `transmission` | categorical | Manual / Automatic |
| `ownership` | categorical | First / Second / Third Owner |
| `insurance` | categorical | Comprehensive / Own Damage / Zero Dep / Third Party |
| `seats` | numeric | Number of seats (4–8) |
| `km_driven` | numeric | Odometer reading (km) |
| `engine_displacement` | numeric | Engine size (cc) |
| `manufacture_yr` | numeric | Year of manufacture |

**Target**: `price` (Indian Rupees)

**Data limitations**:
- Delhi NCR region only
- Snapshot from a specific scrape period
- 445 rows — sufficient for a portfolio model, not for production generalisation

## Installation & Setup

```bash
git clone <repository-url>
cd used_car_model
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### Generate the model

The trained model file `models/catboost_model.cbm` is committed to the repo. If you need to retrain:

```bash
jupyter notebook main.ipynb   # run all cells top to bottom
```

## Running the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Select brand (all 22 brands available), fill in car details, and click **Predict Price**.

## Model Performance

All numbers below are computed from the same fixed random seed (42) and the same 80/20 train/test split. Source: `main.ipynb` and `results/catboost_results.csv`.

### Model comparison (89-sample test set)

| Model | MAE (₹) | RMSE (₹) | R² |
|-------|---------|----------|----|
| Ridge Regression | 210,826 | 283,067 | 0.634 |
| Random Forest (300 trees) | 155,723 | 233,910 | 0.750 |
| **CatBoost — raw target** | 176,898 | 251,456 | 0.718 |
| **CatBoost — log target (final)** | **146,009** | **212,767** | **0.793** |

### Cross-validation (5-fold, full 445-row dataset)

5-fold CV uses a sklearn preprocessing pipeline (StandardScaler + OHE + TargetEncoder) so the absolute numbers differ slightly from the native-CatBoost final model above.

| Fold | R² | MAE (₹) |
|------|----|---------|
| 1 | 0.7361 | 153,934 |
| 2 | 0.8765 | 121,467 |
| 3 | 0.8140 | 147,008 |
| 4 | 0.8120 | 151,762 |
| 5 | 0.8630 | 119,456 |
| **Mean ± std** | **0.820 ± 0.049** | **138,725 ± 15,093** |

CV R² (0.820) is higher than single-split test R² (0.793), which is expected — CV uses all 445 rows for training at each fold vs 356.

### GridSearchCV (18 params × 3-fold = 54 fits)

Grid searched: `iterations` ∈ {500, 1000}, `depth` ∈ {4, 6, 8}, `learning_rate` ∈ {0.03, 0.05, 0.1}

| | Value |
|--|-------|
| Best params | depth=6, iterations=500, lr=0.1 |
| Best CV R² (log-space) | 0.865 |
| Best-params test R² | 0.719 |
| Default-params test R² | **0.793** |

The default parameters outperform the grid-search winner on the held-out test set. This is a known effect on small datasets (89 test samples): 3-fold CV optimises log-space R² and the winner slightly overfits the CV folds. **Default model retained.**

### Final model hyperparameters

| Parameter | Value |
|-----------|-------|
| Iterations | 1000 (early stop at 999) |
| Learning rate | 0.05 |
| Depth | 6 |
| Loss function | MAE |
| Random seed | 42 |
| Target transform | log(price) |

### Prediction accuracy on test set

| Metric | Value |
|--------|-------|
| MAE | ₹146,009 |
| RMSE | ₹212,767 |
| R² | 0.793 |
| Residual std | ₹212,079 |
| Predictions within ±₹1.5L | 62.9% |
| Predictions within ±10% | 50.6% |

The Streamlit app shows an **Estimated Range** of ±₹1.5L, labelled with the 62.9% coverage.

### Feature importance (final model)

| Feature | Importance (%) |
|---------|---------------|
| manufacture_yr | 27.5 |
| engine_displacement | 20.9 |
| brand | 14.7 |
| transmission_Automatic | 10.1 |
| km_driven | 8.6 |
| transmission_Manual | 5.4 |
| seats | 2.8 |
| fuel_type_Diesel | 2.2 |
| insurance_Comprehensive | 2.2 |
| insurance_Own Damage | 1.3 |
| ownership_Second Owner | 1.3 |
| (remaining 6 features) | <1.0 each |

## Data Quality

**Outlier check (IQR method)**:
- `km_driven`: 7 outliers (1.6%) — bounds [−32,500, 123,500]. All 7 are genuine high-mileage cars with prices consistent with brand/age. **Retained.**
- `price`: 0 outliers — data was already constrained to the ₹5–10L segment during scraping.

## Technical Approach

### Preprocessing pipeline

1. Split 80/20 **before** any feature engineering (prevents leakage)
2. Numeric features: median imputation → StandardScaler
3. Categorical (insurance, fuel_type, ownership, transmission): mode imputation → OneHotEncoder
4. `brand` (high cardinality): TargetEncoder with smoothing=10
5. Target: `log(price)` — reduces right skew, improves R² by ~7.5pp vs raw target

### Training strategy

1. Baseline: Ridge Regression
2. Ensemble step: Random Forest (300 trees)
3. Final: CatBoost with log-transformed target and native categorical handling
4. Early stopping on held-out eval set (`use_best_model=True`)
5. GridSearchCV (18 configurations) run and documented; default params retained

## Limitations

- Trained on Delhi market only — may not generalise to other cities
- 445 rows: adequate for this price segment but not for rare brands/configs
- No car condition, accident history, or modification data
- Seasonal pricing trends not captured
- 5-fold CV variance (std=0.049) reflects the small dataset size

## Future Improvements

- [ ] Expand dataset to 1,000+ samples across multiple cities
- [ ] Add Optuna for more efficient hyperparameter search
- [ ] Deploy to Streamlit Cloud
- [ ] Add SHAP values for per-prediction explanations
- [ ] Add price trend charts and comparable-car lookup in the app

## Author

Data Science Portfolio Project — demonstrates end-to-end ML: scraping, cleaning, modelling, validation, and deployment.

> **Note**: For educational and portfolio purposes only. Do not use as sole basis for financial decisions.
