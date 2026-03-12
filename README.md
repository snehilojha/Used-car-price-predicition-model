# Used Car Price Prediction Model

A machine learning project that predicts used car prices in the Indian market using data scraped from CarDekho. This end-to-end project includes web scraping, data preprocessing, model training, and a Streamlit web application for real-time predictions.

## Project Overview

This project addresses the challenge of accurately pricing used cars in the Indian automotive market. By leveraging machine learning algorithms, particularly CatBoost, the model helps buyers and sellers make informed decisions about fair market values.

**Key Features:**
- Custom web scraper for real-time data collection from CarDekho
- Comprehensive data cleaning and feature engineering pipeline
- Multiple ML models compared (Ridge, Random Forest, CatBoost)
- Interactive Streamlit web application for price predictions
- Achieves 79.3% R² score with Mean Absolute Error of ₹146,008

## Business Value

- **For Buyers**: Avoid overpaying by getting data-driven price estimates
- **For Sellers**: Set competitive prices based on market trends
- **For Dealers**: Optimize inventory pricing and improve profit margins
- **Market Insights**: Understand depreciation patterns and key price drivers

## Project Structure

```
used_car_model/
├── data/
│   ├── raw_cardekho_used_cars.csv          # Original scraped data
│   └── cleaned_cardekho_used_cars.csv      # Processed dataset
├── models/
│   └── catboost_model.cbm                  # Trained CatBoost model
├── results/
│   └── catboost_results.csv                # Model predictions and errors
├── catboost_info/                          # CatBoost training logs
├── scrapper.ipynb                          # Web scraping notebook
├── main.ipynb                              # Data cleaning, EDA, and modeling
├── app.py                                  # Streamlit web application
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

## Dataset Description

**Source**: CarDekho (used cars priced between ₹5-10 lakh in New Delhi)

**Size**: 445 records after cleaning

**Features**:
- `brand`: Car manufacturer (Maruti, Hyundai, Toyota, Honda, Kia, Mahindra, Tata, BMW, Audi, etc.)
- `fuel_type`: Petrol, Diesel, or CNG
- `transmission`: Manual or Automatic
- `ownership`: First Owner, Second Owner, Third Owner
- `insurance`: Comprehensive, Own Damage, Zero Dep, Third Party
- `seats`: Number of seats (4-8)
- `km_driven`: Odometer reading in kilometers
- `engine_displacement`: Engine size in cc
- `manufacture_yr`: Year the car was manufactured
- `registration_yr`: Year the car was registered

**Target Variable**: `price` (in Indian Rupees)

**Data Limitations**:
- Limited to Delhi NCR region
- Focused on mid-range price segment (₹5-10 lakh)
- Snapshot from a specific time period
- Sample size of 445 may not capture all market variations

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd used_car_model
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import catboost, streamlit, pandas; print('All dependencies installed successfully!')"
```

## Usage

### Running the Web Application

Launch the Streamlit app for interactive price predictions:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

**How to use the app:**
1. Select car brand from dropdown
2. Choose fuel type, transmission, ownership status, and insurance type
3. Enter numeric values: seats, kilometers driven, engine displacement, manufacture year
4. Click "Predict Price" to get the estimated value

### Running the Notebooks

**Data Collection (Web Scraping)**:
```bash
jupyter notebook scrapper.ipynb
```
- Configurable scraping parameters (scroll depth, page limits)
- Outputs to `data/raw_cardekho_used_cars.csv`

**Data Processing & Modeling**:
```bash
jupyter notebook main.ipynb
```
- Complete pipeline from raw data to trained model
- Includes data cleaning, feature engineering, model training, and evaluation

## Model Performance

### Model Comparison

| Model | MAE (₹) | RMSE (₹) | R² Score |
|-------|---------|----------|----------|
| Ridge Regression | 210,826 | 283,067 | 0.633 |
| Random Forest | 155,723 | 233,910 | 0.750 |
| **CatBoost (Final)** | **146,009** | **N/A** | **0.793** |

### Final Model: CatBoost Regressor

**Why CatBoost?**
- Native handling of categorical features (no manual encoding needed)
- Superior performance on small-to-medium tabular datasets
- Built-in regularization prevents overfitting
- Industry standard for pricing and regression problems

**Hyperparameters**:
- Iterations: 1000 (early stopping at 999)
- Learning rate: 0.05
- Depth: 6
- Loss function: MAE
- Random seed: 42

**Performance Metrics**:
- **Mean Absolute Error**: ₹146,009 (~14.6% of average price)
- **R² Score**: 0.793 (explains 79.3% of price variance)
- **Average Prediction Error**: ₹17,088

### Feature Importance

The model identifies these key price drivers (in order of importance):

1. **Manufacture Year** (28.5%) - Newer cars command higher prices
2. **Engine Displacement** (19.9%) - Larger engines increase value
3. **Transmission** (14.6%) - Automatic transmission premium
4. **Brand** (13.3%) - Brand reputation affects pricing
5. **Kilometers Driven** (7.1%) - Higher mileage reduces value
6. **Fuel Type** (5.3%) - Diesel vs. Petrol preference
7. **Insurance** (5.0%) - Comprehensive coverage adds value
8. **Ownership** (4.5%) - First owner cars are more valuable
9. **Seats** (1.9%) - Minor impact on price

## Technical Approach

### Data Preprocessing Pipeline

1. **Data Cleaning**:
   - Removed year prefix from car names
   - Extracted brand from car model name
   - Converted price strings to numeric (₹ Lakh → actual values)
   - Parsed registration dates to year format
   - Cleaned km_driven and engine_displacement strings
   - Imputed missing insurance values with "Own Damage"

2. **Feature Engineering**:
   - Target encoding for `brand` (high cardinality categorical)
   - One-hot encoding for `insurance`, `fuel_type`, `ownership`, `transmission`
   - Standard scaling for numeric features
   - Log transformation of target variable (improved R² by 3%)

3. **Train-Test Split**:
   - 80/20 split (356 training, 89 test samples)
   - Split performed **before** feature engineering to prevent data leakage
   - Random state: 42 for reproducibility

### Model Training Strategy

1. Started with baseline Ridge regression
2. Improved with Random Forest ensemble
3. Final model: CatBoost with log-transformed target
4. Used early stopping with validation set
5. Saved best model at iteration 999

## Key Insights

1. **Depreciation Pattern**: Cars lose ~15-20% value per year on average
2. **Brand Premium**: Luxury brands (BMW, Audi) maintain higher resale values
3. **Transmission Impact**: Automatic transmission adds ₹50,000-100,000 premium
4. **Mileage Effect**: Every 10,000 km reduces price by approximately ₹15,000
5. **Ownership Matters**: Second owner cars trade at 10-15% discount

## Future Improvements

### Data Collection
- [ ] Expand dataset to 1000+ samples for better generalization
- [ ] Include multiple cities (Mumbai, Bangalore, Chennai)
- [ ] Add temporal data for seasonal pricing trends
- [ ] Scrape additional features (color, accident history, service records)

### Model Enhancements
- [ ] Implement k-fold cross-validation
- [ ] Add confidence intervals to predictions
- [ ] Ensemble multiple models (stacking/blending)
- [ ] Hyperparameter tuning with Optuna or GridSearchCV
- [ ] Add anomaly detection for outlier prices

### Application Features
- [ ] Deploy to Streamlit Cloud or Heroku
- [ ] Add price range predictions (min/max estimates)
- [ ] Include similar car comparisons
- [ ] Visualize feature contributions for each prediction
- [ ] Add historical price trend charts

### Code Quality
- [ ] Refactor notebooks into modular Python scripts
- [ ] Add unit tests for preprocessing functions
- [ ] Implement logging and error handling
- [ ] Create CI/CD pipeline
- [ ] Add API endpoint (FastAPI/Flask)

## Model Validation

**Residual Analysis**: Model errors are approximately normally distributed with slight heteroscedasticity at higher price ranges.

**Limitations**:
- Model trained on Delhi market only (may not generalize to other regions)
- Limited to ₹5-10 lakh price segment
- Does not account for car condition, accident history, or modifications
- Seasonal variations not captured
- Small sample size may lead to overfitting on rare brands/models

## Contributing

Contributions are welcome! Areas for improvement:
- Expand dataset coverage
- Add more sophisticated feature engineering
- Improve model interpretability
- Enhance web application UI/UX

## License

This project is for educational and portfolio purposes.

## Author

**Data Science Portfolio Project**

Demonstrates skills in:
- Web scraping and data collection
- Data cleaning and preprocessing
- Feature engineering
- Machine learning model development
- Model evaluation and selection
- Deployment with Streamlit
- End-to-end ML project execution


**Note**: This model is for educational purposes and should not be used as the sole basis for financial decisions. Always consult with automotive experts and conduct thorough inspections before purchasing used vehicles.
