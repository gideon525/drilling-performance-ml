# ROP Prediction — Model Evaluation

## Dataset
- Dataset: USROP
- Total observations: 198,928
- Training wells: [0, 1, 2, 3, 4]
- Held-out test wells: [5, 6]
- Test observations: 26,399

## Selected Model
- Algorithm: Random Forest Regressor
- Trees: 50
- Maximum depth: 6
- Minimum samples per leaf: 10
- Features: 9

## Held-Out Test Performance
- MAE: 14.34 m/h
- RMSE: 18.23 m/h
- R2: 0.1547
- Baseline MAE: 17.70 m/h

## Key Findings
- Prediction performance varies between wells.
- Low ROP is frequently overpredicted.
- High ROP is frequently underpredicted.
- Hookload has the highest impurity-based feature importance.

## Limitations
- Repeated drilling measurements may affect model behavior.
- Well 3 mud-density units require independent verification.
- Cross-well generalization remains inconsistent.
- Feature importance does not establish causation.
- Bit-health prediction is not validated because labeled
  bit-wear or failure outcomes are unavailable.
- The model is not validated for operational deployment.
