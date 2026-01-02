# Used-car-price-predicition-model

## Project Overview

Problem Statement
Used car prices vary widely due to brand perception, vehicle condition, age, and specifications.
The objective is to build a robust ML model to predict car prices using structured tabular data.

### Why this matters:

- Real-world pricing problem
- Mixed numerical + categorical data
- Non-linear relationships

## Modeling Approach

### Baseline:
- Ridge Regression
- Proper preprocessing
- Used for interpretability and benchmarking

### Final Model
- CatBoost Regressor
- Native handling of categorical variables
- Robust on small datasets
- Avoids leakage from manual target encoding

### Target Transformation
- Log-transform of price to reduce skew
- Inverse transform for final evaluation

### Final Training code

```
from catboost import CatBoostRegressor
import numpy as np

cat_features = [
    'brand', 'fuel_type', 'insurance',
    'ownership', 'transmission'
]

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='MAE',
    random_seed=42,
    verbose=100
)

model.fit(
    X_train,
    np.log(y_train),
    cat_features=cat_features,
    eval_set=(X_test, np.log(y_test)),
    use_best_model=True
)
```

### Evaluation metrics

| Metric |          Reason          |
|:------:|:------------------------:|
| MAE    | Robust to outliers       |
| R^2    | Model explanatory powers |

### Error analysis

```
results = X_test.copy()
results['y_test'] = y_test.values
results['y_pred'] = np.exp(model.predict(X_test))
results['error'] = results['y_pred'] - results['y_test']
```

### Feature Importance

Top features
- Manufacturing year
- Engine displacement
- Transmission

### Insights
- Model achieved low average error (~₹17k) relative to mean car prices
- Performed strongest on mass-market vehicles; higher variance on premium segments
- Manufacturing year,Transmission and Engine displacement emerged as dominant pricing drivers
- Errors increased for rare brand–year combinations and atypical usage patterns
- Results align with real-world car market behavior

