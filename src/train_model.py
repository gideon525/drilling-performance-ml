
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Locate the processed drilling dataset.
project_root = Path(__file__).resolve().parents[1]
data_path = project_root / "data" / "processed" / "usrop_processed.parquet"

# Load the dataset for machine learning.
drilling_data = pd.read_parquet(data_path)

print("\nPHASE 4 — MODEL DEVELOPMENT")
print("Dataset loaded successfully.")
print("Total records:", f"{len(drilling_data):,}")
print("Total wells:", drilling_data["well_id"].nunique())
print("Total columns:", len(drilling_data.columns))


# Define the prediction target.
target_column = "Rate of Penetration m/h"

# Define the initial model input features.
feature_columns = [
    "Measured Depth m",
    "Weight on Bit kkgf",
    "Average Standpipe Pressure kPa",
    "Average Surface Torque kN.m",
    "Average Rotary Speed rpm",
    "Mud Flow In L/min",
    "mud_density_corrected_g_cm3",
    "Diameter mm",
    "Average Hookload kkgf",
    "Hole Depth (TVD) m",
    "USROP Gamma gAPI"
]

# Separate model inputs from the prediction target.
X = drilling_data[feature_columns]
y = drilling_data[target_column]

print("\nSTEP 4.2 — FEATURE AND TARGET SELECTION")
print("Number of input features:", X.shape[1])
print("Number of observations:", X.shape[0])
print("Target variable:", y.name)
print("Missing input values:", X.isna().sum().sum())
print("Missing target values:", y.isna().sum())


# Split by well to prevent the same well appearing in both sets.
train_wells = [0, 1, 2, 3, 4]
test_wells = [5, 6]

train_mask = drilling_data["well_id"].isin(train_wells)
test_mask = drilling_data["well_id"].isin(test_wells)

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

# Verify that training and testing wells do not overlap.
assert not set(train_wells).intersection(test_wells)
assert train_mask.sum() + test_mask.sum() == len(drilling_data)

print("\nSTEP 4.3 — WELL-BASED TRAIN/TEST SPLIT")
print("Training wells:", train_wells)
print("Testing wells:", test_wells)
print("Training records:", f"{len(X_train):,}")
print("Testing records:", f"{len(X_test):,}")
print("Number of features:", X_train.shape[1])


from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Train a baseline that always predicts the training mean ROP.
baseline_model = DummyRegressor(strategy="mean")
baseline_model.fit(X_train, y_train)

# Evaluate the baseline on training data only.
baseline_train_predictions = baseline_model.predict(X_train)

baseline_mae = mean_absolute_error(
    y_train, baseline_train_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(y_train, baseline_train_predictions)
)

print("\nSTEP 4.4 — BASELINE ROP MODEL")
print(f"Baseline predicted ROP: {baseline_model.constant_[0, 0]:.2f} m/h")
print(f"Training MAE: {baseline_mae:.2f} m/h")
print(f"Training RMSE: {baseline_rmse:.2f} m/h")


from sklearn.linear_model import LinearRegression

# Train a linear regression model using Wells 0–4.
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Evaluate the model on training observations.
linear_train_predictions = linear_model.predict(X_train)

linear_mae = mean_absolute_error(
    y_train, linear_train_predictions
)

linear_rmse = np.sqrt(
    mean_squared_error(y_train, linear_train_predictions)
)

print("\nSTEP 4.5 — LINEAR REGRESSION MODEL")
print(f"Training MAE: {linear_mae:.2f} m/h")
print(f"Training RMSE: {linear_rmse:.2f} m/h")

print("\nCOMPARISON WITH BASELINE")
print(f"Baseline MAE: {baseline_mae:.2f} m/h")
print(f"Linear Regression MAE: {linear_mae:.2f} m/h")


from sklearn.metrics import r2_score

# Calculate training R² for both models.
baseline_r2 = r2_score(
    y_train, baseline_train_predictions
)

linear_r2 = r2_score(
    y_train, linear_train_predictions
)

print("\nSTEP 4.6 — MODEL R-SQUARED COMPARISON")
print(f"Baseline Training R²: {baseline_r2:.4f}")
print(f"Linear Regression Training R²: {linear_r2:.4f}")


from sklearn.ensemble import RandomForestRegressor

# Train an initial nonlinear ROP prediction model.
rf_model = RandomForestRegressor(
    n_estimators=50,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Evaluate Random Forest on training observations.
rf_train_predictions = rf_model.predict(X_train)

rf_mae = mean_absolute_error(
    y_train, rf_train_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(y_train, rf_train_predictions)
)

rf_r2 = r2_score(
    y_train, rf_train_predictions
)

print("\nSTEP 4.7 — RANDOM FOREST MODEL")
print(f"Training MAE: {rf_mae:.2f} m/h")
print(f"Training RMSE: {rf_rmse:.2f} m/h")
print(f"Training R²: {rf_r2:.4f}")


# Validate Random Forest on an unseen training-group well.
validation_train_mask = drilling_data["well_id"].isin([0, 1, 2, 3])
validation_mask = drilling_data["well_id"] == 4

X_validation_train = X.loc[validation_train_mask]
y_validation_train = y.loc[validation_train_mask]

X_validation = X.loc[validation_mask]
y_validation = y.loc[validation_mask]

# Train a separate model without using Well 4.
rf_validation_model = RandomForestRegressor(
    n_estimators=50,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf_validation_model.fit(
    X_validation_train,
    y_validation_train
)

validation_predictions = rf_validation_model.predict(
    X_validation
)

validation_mae = mean_absolute_error(
    y_validation,
    validation_predictions
)

validation_rmse = np.sqrt(
    mean_squared_error(y_validation, validation_predictions)
)

validation_r2 = r2_score(
    y_validation,
    validation_predictions
)

print("\nSTEP 4.8 — UNSEEN WELL VALIDATION")
print("Training wells: [0, 1, 2, 3]")
print("Validation well: 4")
print(f"Validation MAE: {validation_mae:.2f} m/h")
print(f"Validation RMSE: {validation_rmse:.2f} m/h")
print(f"Validation R²: {validation_r2:.4f}")


# Validate Random Forest on an unseen training-group well.
validation_train_mask = drilling_data["well_id"].isin([0, 1, 2, 3])
validation_mask = drilling_data["well_id"] == 4

X_validation_train = X.loc[validation_train_mask]
y_validation_train = y.loc[validation_train_mask]

X_validation = X.loc[validation_mask]
y_validation = y.loc[validation_mask]

# Train a separate model without using Well 4.
rf_validation_model = RandomForestRegressor(
    n_estimators=50,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf_validation_model.fit(
    X_validation_train,
    y_validation_train
)

validation_predictions = rf_validation_model.predict(
    X_validation
)

validation_mae = mean_absolute_error(
    y_validation,
    validation_predictions
)

validation_rmse = np.sqrt(
    mean_squared_error(y_validation, validation_predictions)
)

validation_r2 = r2_score(
    y_validation,
    validation_predictions
)

print("\nSTEP 4.8 — UNSEEN WELL VALIDATION")
print("Training wells: [0, 1, 2, 3]")
print("Validation well: 4")
print(f"Validation MAE: {validation_mae:.2f} m/h")
print(f"Validation RMSE: {validation_rmse:.2f} m/h")
print(f"Validation R²: {validation_r2:.4f}")


# Compare Linear Regression on the same unseen validation well.
linear_validation_model = LinearRegression()

linear_validation_model.fit(
    X_validation_train,
    y_validation_train
)

linear_validation_predictions = linear_validation_model.predict(
    X_validation
)

linear_validation_mae = mean_absolute_error(
    y_validation,
    linear_validation_predictions
)

linear_validation_rmse = np.sqrt(
    mean_squared_error(
        y_validation,
        linear_validation_predictions
    )
)

linear_validation_r2 = r2_score(
    y_validation,
    linear_validation_predictions
)

print("\nSTEP 4.9 — WELL 4 MODEL COMPARISON")

print(
    f"Random Forest MAE: {validation_mae:.2f} m/h"
)
print(
    f"Linear Regression MAE: {linear_validation_mae:.2f} m/h"
)

print(
    f"Random Forest R²: {validation_r2:.4f}"
)
print(
    f"Linear Regression R²: {linear_validation_r2:.4f}"
)


from sklearn.model_selection import LeaveOneGroupOut

# Prepare well-based cross-validation using training wells only.
logo = LeaveOneGroupOut()

cv_groups = drilling_data.loc[
    train_mask, "well_id"
]

print("\nSTEP 4.10 — CROSS-VALIDATION SETUP")

for fold_number, (train_idx, val_idx) in enumerate(
    logo.split(X_train, y_train, groups=cv_groups),
    start=1
):
    fold_train_wells = sorted(
        cv_groups.iloc[train_idx].unique().tolist()
    )

    fold_val_wells = sorted(
        cv_groups.iloc[val_idx].unique().tolist()
    )

    print(
        f"Fold {fold_number}: "
        f"Train {fold_train_wells} | "
        f"Validate {fold_val_wells}"
    )

# Evaluate Random Forest across all five training wells.
cv_results = []

print("\nSTEP 4.11 — RANDOM FOREST CROSS-VALIDATION")

for fold_number, (train_idx, val_idx) in enumerate(
    logo.split(X_train, y_train, groups=cv_groups),
    start=1
):
    X_fold_train = X_train.iloc[train_idx]
    y_fold_train = y_train.iloc[train_idx]

    X_fold_val = X_train.iloc[val_idx]
    y_fold_val = y_train.iloc[val_idx]

    validation_well = int(
        cv_groups.iloc[val_idx].iloc[0]
    )

    # Fit a new model without the validation well.
    fold_model = RandomForestRegressor(
        n_estimators=50,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    fold_model.fit(X_fold_train, y_fold_train)
    fold_predictions = fold_model.predict(X_fold_val)

    fold_mae = mean_absolute_error(
        y_fold_val, fold_predictions
    )
    fold_rmse = np.sqrt(
        mean_squared_error(y_fold_val, fold_predictions)
    )
    fold_r2 = r2_score(
        y_fold_val, fold_predictions
    )

    cv_results.append({
        "well_id": validation_well,
        "mae": fold_mae,
        "rmse": fold_rmse,
        "r2": fold_r2
    })

    print(
        f"Well {validation_well}: "
        f"MAE={fold_mae:.2f}, "
        f"RMSE={fold_rmse:.2f}, "
        f"R²={fold_r2:.4f}"
    )

cv_results_df = pd.DataFrame(cv_results)

print("\nCROSS-VALIDATION SUMMARY")
print(f"Mean MAE: {cv_results_df['mae'].mean():.2f} m/h")
print(f"Mean RMSE: {cv_results_df['rmse'].mean():.2f} m/h")
print(f"Mean R²: {cv_results_df['r2'].mean():.4f}")


# Compare validation errors and actual ROP by well.
well_rop_summary = (
    drilling_data.loc[train_mask]
    .groupby("well_id")["Rate of Penetration m/h"]
    .agg(["mean", "std", "min", "max"])
    .reset_index()
)

well_rop_summary.columns = [
    "well_id",
    "mean_rop",
    "std_rop",
    "min_rop",
    "max_rop"
]

well_diagnostics = cv_results_df.merge(
    well_rop_summary,
    on="well_id"
)

well_diagnostics["mae_pct_mean_rop"] = (
    well_diagnostics["mae"]
    / well_diagnostics["mean_rop"]
    * 100
)

print("\nSTEP 4.12 — WELL-LEVEL ERROR DIAGNOSTICS")

print(
    well_diagnostics[
        [
            "well_id",
            "mean_rop",
            "std_rop",
            "mae",
            "rmse",
            "r2",
            "mae_pct_mean_rop"
        ]
    ].round(2).to_string(index=False)
)

# Compare average drilling conditions across training wells.
diagnostic_features = [
    "Weight on Bit kkgf",
    "Average Rotary Speed rpm",
    "Average Standpipe Pressure kPa",
    "USROP Gamma gAPI"
]

feature_summary = (
    drilling_data.loc[train_mask]
    .groupby("well_id")[diagnostic_features]
    .mean()
)

feature_summary.columns = [
    "mean_wob",
    "mean_rpm",
    "mean_pressure",
    "mean_gamma"
]

print("\nSTEP 4.13 — FEATURE DISTRIBUTION ANALYSIS")
print(feature_summary.round(2).to_string())


# Measure feature values outside each fold's training range.
range_results = []

for train_idx, val_idx in logo.split(
    X_train, y_train, groups=cv_groups
):
    fold_train = X_train.iloc[train_idx]
    fold_val = X_train.iloc[val_idx]

    validation_well = int(
        cv_groups.iloc[val_idx].iloc[0]
    )

    for feature in diagnostic_features:
        train_min = fold_train[feature].min()
        train_max = fold_train[feature].max()

        outside_range = (
            (fold_val[feature] < train_min)
            | (fold_val[feature] > train_max)
        )

        range_results.append({
            "well_id": validation_well,
            "feature": feature,
            "outside_pct": outside_range.mean() * 100
        })

range_df = pd.DataFrame(range_results)

range_table = range_df.pivot(
    index="well_id",
    columns="feature",
    values="outside_pct"
)

range_table = range_table.rename(columns={
    "Weight on Bit kkgf": "WOB",
    "Average Rotary Speed rpm": "RPM",
    "Average Standpipe Pressure kPa": "Pressure",
    "USROP Gamma gAPI": "Gamma"
})

print("\nSTEP 4.14 — FEATURE RANGE DIAGNOSTICS")
print(range_table.round(2).to_string())


# Collect out-of-fold predictions for each training well.
cv_prediction_results = []

for train_idx, val_idx in logo.split(
    X_train, y_train, groups=cv_groups
):
    fold_model = RandomForestRegressor(
        n_estimators=50,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    fold_model.fit(
        X_train.iloc[train_idx],
        y_train.iloc[train_idx]
    )

    fold_predictions = fold_model.predict(
        X_train.iloc[val_idx]
    )

    validation_well = int(
        cv_groups.iloc[val_idx].iloc[0]
    )

    cv_prediction_results.append({
        "well_id": validation_well,
        "actual_mean_rop": y_train.iloc[val_idx].mean(),
        "predicted_mean_rop": fold_predictions.mean()
    })

bias_df = pd.DataFrame(cv_prediction_results)

# Positive bias means the model overpredicts ROP.
bias_df["prediction_bias"] = (
    bias_df["predicted_mean_rop"]
    - bias_df["actual_mean_rop"]
)

print("\nSTEP 4.15 — CROSS-WELL PREDICTION BIAS")
print(bias_df.round(2).to_string(index=False))


# Evaluate a mean-ROP baseline on unseen wells.
baseline_cv_results = []

for train_idx, val_idx in logo.split(
    X_train, y_train, groups=cv_groups
):
    fold_baseline = DummyRegressor(strategy="mean")

    fold_baseline.fit(
        X_train.iloc[train_idx],
        y_train.iloc[train_idx]
    )

    baseline_predictions = fold_baseline.predict(
        X_train.iloc[val_idx]
    )

    validation_well = int(
        cv_groups.iloc[val_idx].iloc[0]
    )

    baseline_cv_results.append({
        "well_id": validation_well,
        "baseline_mae": mean_absolute_error(
            y_train.iloc[val_idx],
            baseline_predictions
        ),
        "baseline_rmse": np.sqrt(
            mean_squared_error(
                y_train.iloc[val_idx],
                baseline_predictions
            )
        ),
        "baseline_r2": r2_score(
            y_train.iloc[val_idx],
            baseline_predictions
        )
    })

baseline_cv_df = pd.DataFrame(baseline_cv_results)

comparison_df = cv_results_df.merge(
    baseline_cv_df,
    on="well_id"
)

print("\nSTEP 4.16 — CROSS-WELL BASELINE COMPARISON")

print(
    comparison_df[
        ["well_id", "mae", "baseline_mae",
         "r2", "baseline_r2"]
    ].round(2).to_string(index=False)
)

print("\nMEAN VALIDATION MAE")
print(f"Random Forest: {comparison_df['mae'].mean():.2f} m/h")
print(f"Baseline: {comparison_df['baseline_mae'].mean():.2f} m/h")


# Test Random Forest without absolute depth features.
reduced_features = [
    feature for feature in feature_columns
    if feature not in [
        "Measured Depth m",
        "Hole Depth (TVD) m"
    ]
]

X_train_reduced = X_train[reduced_features]

reduced_cv_results = []

print("\nSTEP 4.17 — RANDOM FOREST WITHOUT DEPTH FEATURES")

for train_idx, val_idx in logo.split(
    X_train_reduced, y_train, groups=cv_groups
):
    reduced_model = RandomForestRegressor(
        n_estimators=50,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    reduced_model.fit(
        X_train_reduced.iloc[train_idx],
        y_train.iloc[train_idx]
    )

    predictions = reduced_model.predict(
        X_train_reduced.iloc[val_idx]
    )

    validation_well = int(
        cv_groups.iloc[val_idx].iloc[0]
    )

    fold_mae = mean_absolute_error(
        y_train.iloc[val_idx],
        predictions
    )

    reduced_cv_results.append({
        "well_id": validation_well,
        "reduced_mae": fold_mae
    })

    print(
        f"Well {validation_well}: "
        f"MAE={fold_mae:.2f} m/h"
    )

reduced_cv_df = pd.DataFrame(reduced_cv_results)

print("\nSTEP 4.17 — MODEL COMPARISON")
print(f"Original Random Forest MAE: {cv_results_df['mae'].mean():.2f} m/h")
print(f"Without Depth Features MAE: {reduced_cv_df['reduced_mae'].mean():.2f} m/h")
print(f"Baseline MAE: {baseline_cv_df['baseline_mae'].mean():.2f} m/h")


# Test a simpler Random Forest without absolute depth.
simple_cv_results = []

print("\nSTEP 4.18 — SIMPLER RANDOM FOREST")

for train_idx, val_idx in logo.split(
    X_train_reduced, y_train, groups=cv_groups
):
    simple_model = RandomForestRegressor(
        n_estimators=50,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )

    simple_model.fit(
        X_train_reduced.iloc[train_idx],
        y_train.iloc[train_idx]
    )

    predictions = simple_model.predict(
        X_train_reduced.iloc[val_idx]
    )

    validation_well = int(
        cv_groups.iloc[val_idx].iloc[0]
    )

    fold_mae = mean_absolute_error(
        y_train.iloc[val_idx],
        predictions
    )

    simple_cv_results.append({
        "well_id": validation_well,
        "simple_mae": fold_mae
    })

    print(
        f"Well {validation_well}: "
        f"MAE={fold_mae:.2f} m/h"
    )

simple_cv_df = pd.DataFrame(simple_cv_results)

print("\nSTEP 4.18 — MODEL COMPARISON")
print(f"Baseline: {baseline_cv_df['baseline_mae'].mean():.2f} m/h")
print(f"Original RF: {cv_results_df['mae'].mean():.2f} m/h")
print(f"RF without depth: {reduced_cv_df['reduced_mae'].mean():.2f} m/h")
print(f"Simpler RF: {simple_cv_df['simple_mae'].mean():.2f} m/h")


# Compare model errors on each validation well.
model_comparison = (
    baseline_cv_df[["well_id", "baseline_mae"]]
    .merge(
        cv_results_df[["well_id", "mae"]],
        on="well_id"
    )
    .merge(
        reduced_cv_df,
        on="well_id"
    )
    .merge(
        simple_cv_df,
        on="well_id"
    )
)

model_comparison = model_comparison.rename(columns={
    "mae": "original_rf_mae",
    "reduced_mae": "no_depth_mae",
    "simple_mae": "simple_rf_mae"
})

# Positive improvement means the simpler model beats baseline.
model_comparison["improvement_pct"] = (
    (
        model_comparison["baseline_mae"]
        - model_comparison["simple_rf_mae"]
    )
    / model_comparison["baseline_mae"]
    * 100
)

print("\nSTEP 4.19 — PER-WELL MODEL COMPARISON")
print(model_comparison.round(2).to_string(index=False))

print("\nWELLS WHERE SIMPLE RF BEATS BASELINE")
print(
    (
        model_comparison["simple_rf_mae"]
        < model_comparison["baseline_mae"]
    ).sum()
)


# Fit the simpler model on all five training wells.
final_candidate_rf = RandomForestRegressor(
    n_estimators=50,
    max_depth=6,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

final_candidate_rf.fit(
    X_train_reduced,
    y_train
)

# Extract impurity-based feature importance.
feature_importance = pd.DataFrame({
    "feature": reduced_features,
    "importance": final_candidate_rf.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
)

print("\nSTEP 4.20 — RANDOM FOREST FEATURE IMPORTANCE")
print(
    feature_importance.round(4).to_string(index=False)
)


# Test the simpler Random Forest without Hookload.
no_hookload_features = [
    feature for feature in reduced_features
    if feature != "Average Hookload kkgf"
]

X_train_no_hookload = X_train[no_hookload_features]

no_hookload_results = []

print("\nSTEP 4.21 — HOOKLOAD ABLATION TEST")

for train_idx, val_idx in logo.split(
    X_train_no_hookload, y_train, groups=cv_groups
):
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train_no_hookload.iloc[train_idx],
        y_train.iloc[train_idx]
    )

    predictions = model.predict(
        X_train_no_hookload.iloc[val_idx]
    )

    validation_well = int(
        cv_groups.iloc[val_idx].iloc[0]
    )

    fold_mae = mean_absolute_error(
        y_train.iloc[val_idx],
        predictions
    )

    no_hookload_results.append({
        "well_id": validation_well,
        "no_hookload_mae": fold_mae
    })

    print(
        f"Well {validation_well}: "
        f"MAE={fold_mae:.2f} m/h"
    )

no_hookload_df = pd.DataFrame(no_hookload_results)

print("\nSTEP 4.21 — MODEL COMPARISON")
print(
    f"Baseline MAE: "
    f"{baseline_cv_df['baseline_mae'].mean():.2f} m/h"
)
print(
    f"Simpler RF MAE: "
    f"{simple_cv_df['simple_mae'].mean():.2f} m/h"
)
print(
    f"RF without Hookload MAE: "
    f"{no_hookload_df['no_hookload_mae'].mean():.2f} m/h"
)


# Examine Hookload–ROP correlation separately for each well.
hookload_correlations = (
    drilling_data.loc[train_mask]
    .groupby("well_id")
    .apply(
        lambda group: group[
            "Average Hookload kkgf"
        ].corr(
            group["Rate of Penetration m/h"]
        ),
        include_groups=False
    )
)

print("\nSTEP 4.22 — HOOKLOAD AND ROP CORRELATION")
print(hookload_correlations.round(4).to_string())


# Compare original and corrected mud-density values by well.
mud_density_summary = (
    drilling_data.loc[train_mask]
    .groupby("well_id")
    .agg(
        original_min=("mud_density_original", "min"),
        original_mean=("mud_density_original", "mean"),
        original_max=("mud_density_original", "max"),
        corrected_min=("mud_density_corrected_g_cm3", "min"),
        corrected_mean=("mud_density_corrected_g_cm3", "mean"),
        corrected_max=("mud_density_corrected_g_cm3", "max")
    )
)

print("\nSTEP 4.23 — MUD DENSITY UNIT DIAGNOSTICS")
print(mud_density_summary.round(3).to_string())


# Compare original versus corrected mud-density assumptions.
original_density_features = [
    "mud_density_original"
    if feature == "mud_density_corrected_g_cm3"
    else feature
    for feature in reduced_features
]

X_train_original_density = drilling_data.loc[
    train_mask, original_density_features
]

original_density_mae = []

print("\nSTEP 4.24 — MUD DENSITY SENSITIVITY TEST")

for train_idx, val_idx in logo.split(
    X_train_original_density, y_train, groups=cv_groups
):
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train_original_density.iloc[train_idx],
        y_train.iloc[train_idx]
    )

    predictions = model.predict(
        X_train_original_density.iloc[val_idx]
    )

    validation_well = int(
        cv_groups.iloc[val_idx].iloc[0]
    )

    fold_mae = mean_absolute_error(
        y_train.iloc[val_idx],
        predictions
    )

    original_density_mae.append(fold_mae)

    print(
        f"Well {validation_well}: "
        f"MAE={fold_mae:.2f} m/h"
    )

print("\nSTEP 4.24 — SENSITIVITY COMPARISON")
print(
    f"Corrected Density MAE: "
    f"{simple_cv_df['simple_mae'].mean():.2f} m/h"
)
print(
    f"Original Density MAE: "
    f"{np.mean(original_density_mae):.2f} m/h"
)


# Consolidate cross-validation results before final testing.
final_cv_comparison = model_comparison.copy()

final_cv_comparison["original_density_mae"] = (
    original_density_mae
)

final_cv_comparison["no_hookload_mae"] = (
    no_hookload_df["no_hookload_mae"].values
)

report_path = (
    project_root
    / "reports"
    / "phase4_model_comparison.csv"
)

report_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

final_cv_comparison.to_csv(
    report_path,
    index=False
)

print("\nSTEP 4.25 — MODEL SELECTION REPORT")
print("Report saved:", report_path)
print("Number of validation wells:", len(final_cv_comparison))
print("Selected model: Simpler Random Forest")
print("Selected features:", len(reduced_features))
print("Selected model CV MAE: 15.28 m/h")


import joblib

# Save the selected model and its feature configuration.
model_path = project_root / "models" / "rop_random_forest.joblib"

model_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

model_artifact = {
    "model": final_candidate_rf,
    "features": reduced_features,
    "training_wells": train_wells,
    "test_wells": test_wells,
    "mud_density_assumption": "Well 3 provisionally converted from ppg to g/cm3",
    "model_version": "1.0"
}

joblib.dump(model_artifact, model_path)

# Verify that the saved model can be loaded.
loaded_artifact = joblib.load(model_path)

assert loaded_artifact["features"] == reduced_features
assert loaded_artifact["model"].n_features_in_ == 9

print("\nSTEP 4.26 — MODEL SERIALIZATION")
print("Model saved:", model_path)
print("Model version:", loaded_artifact["model_version"])
print("Number of features:", len(loaded_artifact["features"]))
print("Training wells:", loaded_artifact["training_wells"])
print("Test wells:", loaded_artifact["test_wells"])
print("Model verification: PASSED")


# Evaluate the saved model on the untouched test wells.
saved_model = loaded_artifact["model"]
saved_features = loaded_artifact["features"]

X_final_test = X_test[saved_features]

test_predictions = saved_model.predict(X_final_test)
baseline_test_predictions = baseline_model.predict(X_test)

print("\nSTEP 4.27 — FINAL HELD-OUT TEST EVALUATION")

for well in test_wells:
    well_mask = drilling_data.loc[test_mask, "well_id"] == well

    actual = y_test.loc[well_mask]
    predicted = test_predictions[well_mask.to_numpy()]
    baseline_predicted = baseline_test_predictions[
        well_mask.to_numpy()
    ]

    print(f"\nWell {well}")
    print(f"Observations: {len(actual):,}")
    print(f"Actual Mean ROP: {actual.mean():.2f} m/h")
    print(f"Predicted Mean ROP: {predicted.mean():.2f} m/h")

    print(
        f"Random Forest MAE: "
        f"{mean_absolute_error(actual, predicted):.2f} m/h"
    )
    print(
        f"Baseline MAE: "
        f"{mean_absolute_error(actual, baseline_predicted):.2f} m/h"
    )
    print(
        f"Random Forest R²: "
        f"{r2_score(actual, predicted):.4f}"
    )

print("\nCOMBINED TEST PERFORMANCE")

print(
    f"Random Forest MAE: "
    f"{mean_absolute_error(y_test, test_predictions):.2f} m/h"
)
print(
    f"Random Forest RMSE: "
    f"{np.sqrt(mean_squared_error(y_test, test_predictions)):.2f} m/h"
)
print(
    f"Random Forest R²: "
    f"{r2_score(y_test, test_predictions):.4f}"
)
print(
    f"Baseline MAE: "
    f"{mean_absolute_error(y_test, baseline_test_predictions):.2f} m/h"
)


# Save held-out predictions for subsequent error analysis.
test_results = drilling_data.loc[
    test_mask, ["well_id", target_column]
].copy()

test_results["predicted_rop"] = test_predictions
test_results["baseline_predicted_rop"] = baseline_test_predictions

test_results["absolute_error"] = (
    test_results[target_column] - test_results["predicted_rop"]
).abs()

test_results["prediction_error"] = (
    test_results["predicted_rop"] - test_results[target_column]
)

output_path = (
    project_root / "reports" / "final_test_predictions.csv"
)

test_results.to_csv(output_path, index=False)

print("\nSTEP 4.28 — FINAL TEST RESULTS SAVED")
print("Report:", output_path)
print("Observations:", len(test_results))
print("Wells:", sorted(test_results["well_id"].unique()))
print("Mean Absolute Error:", round(test_results["absolute_error"].mean(), 2))


# Examine prediction bias for each held-out well.
bias_summary = (
    test_results.groupby("well_id")
    .agg(
        observations=(target_column, "size"),
        actual_mean_rop=(target_column, "mean"),
        predicted_mean_rop=("predicted_rop", "mean"),
        mean_absolute_error=("absolute_error", "mean"),
        mean_prediction_error=("prediction_error", "mean")
    )
)

print("\nSTEP 5.1 — PREDICTION BIAS BY WELL")
print(bias_summary.round(2).to_string())


# Group held-out predictions by actual drilling ROP.
rop_bins = [0, 10, 20, 40, 60, float("inf")]

rop_labels = [
    "0-10",
    "10-20",
    "20-40",
    "40-60",
    "60+"
]

test_results["rop_range"] = pd.cut(
    test_results[target_column],
    bins=rop_bins,
    labels=rop_labels,
    include_lowest=True
)

rop_error_summary = (
    test_results.groupby(
        "rop_range",
        observed=True
    )
    .agg(
        observations=(target_column, "size"),
        actual_mean_rop=(target_column, "mean"),
        predicted_mean_rop=("predicted_rop", "mean"),
        mean_absolute_error=("absolute_error", "mean"),
        prediction_bias=("prediction_error", "mean")
    )
)

print("\nSTEP 5.2 — ERROR ANALYSIS BY ROP RANGE")
print(rop_error_summary.round(2).to_string())


# Compare ROP-range errors separately for each test well.
well_rop_errors = (
    test_results.groupby(
        ["well_id", "rop_range"],
        observed=True
    )
    .agg(
        observations=(target_column, "size"),
        mean_absolute_error=("absolute_error", "mean"),
        prediction_bias=("prediction_error", "mean")
    )
    .reset_index()
)

print("\nSTEP 5.3 — ROP ERRORS BY WELL AND RANGE")
print(well_rop_errors.round(2).to_string(index=False))


# Visualize actual versus predicted ROP for the held-out wells.
plt.figure(figsize=(9, 7))

for well in test_wells:
    well_data = test_results[
        test_results["well_id"] == well
    ]

    plt.scatter(
        well_data[target_column],
        well_data["predicted_rop"],
        s=6,
        alpha=0.15,
        label=f"Well {well}"
    )

plt.plot(
    [0, 100],
    [0, 100],
    "k--",
    linewidth=1.5,
    label="Perfect prediction"
)

plt.xlabel("Actual ROP (m/h)")
plt.ylabel("Predicted ROP (m/h)")
plt.title("Actual vs Predicted ROP — Held-Out Wells")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

plot_path = (
    project_root
    / "reports"
    / "actual_vs_predicted_rop.png"
)

plt.savefig(plot_path, dpi=200)
plt.close()

print("\nSTEP 5.4 — ACTUAL VS PREDICTED PLOT")
print("Plot saved:", plot_path)


# Visualize prediction residuals for both held-out wells.
plt.figure(figsize=(9, 6))

for well in test_wells:
    well_data = test_results[
        test_results["well_id"] == well
    ]

    plt.scatter(
        well_data[target_column],
        well_data["prediction_error"],
        s=6,
        alpha=0.15,
        label=f"Well {well}"
    )

plt.axhline(
    y=0,
    color="black",
    linestyle="--",
    linewidth=1.5
)

plt.xlabel("Actual ROP (m/h)")
plt.ylabel("Prediction Error (m/h)")
plt.title("ROP Prediction Residuals — Held-Out Wells")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

residual_plot_path = (
    project_root
    / "reports"
    / "rop_prediction_residuals.png"
)

plt.savefig(residual_plot_path, dpi=200)
plt.close()

print("\nSTEP 5.5 — RESIDUAL ERROR VISUALIZATION")
print("Plot saved:", residual_plot_path)


# Visualize feature importance from the frozen Random Forest.
feature_importance = pd.Series(
    saved_model.feature_importances_,
    index=saved_features
).sort_values()

plt.figure(figsize=(10, 6))

feature_importance.plot(
    kind="barh"
)

plt.xlabel("Random Forest Feature Importance")
plt.ylabel("Drilling Parameter")
plt.title("ROP Prediction — Feature Importance")
plt.tight_layout()

importance_plot_path = (
    project_root
    / "reports"
    / "rop_feature_importance.png"
)

plt.savefig(importance_plot_path, dpi=200)
plt.close()

print("\nSTEP 5.6 — FEATURE IMPORTANCE")
print(feature_importance.sort_values(ascending=False).round(4))
print("\nPlot saved:", importance_plot_path)


# Export model feature importance for reporting.
importance_report = (
    feature_importance
    .sort_values(ascending=False)
    .rename("importance")
    .reset_index()
    .rename(columns={"index": "feature"})
)

importance_report["importance_percent"] = (
    importance_report["importance"] * 100
)

importance_csv_path = (
    project_root
    / "reports"
    / "rop_feature_importance.csv"
)

importance_report.to_csv(
    importance_csv_path,
    index=False
)

print("\nSTEP 5.7 — FEATURE IMPORTANCE REPORT")
print("Report saved:", importance_csv_path)
print("Features exported:", len(importance_report))
print(
    "Total importance:",
    round(importance_report["importance"].sum(), 4)
)


# Save a concise evaluation summary for the project report.
evaluation_summary = f"""# ROP Prediction — Model Evaluation

## Dataset
- Dataset: USROP
- Total observations: {len(drilling_data):,}
- Training wells: {train_wells}
- Held-out test wells: {test_wells}
- Test observations: {len(y_test):,}

## Selected Model
- Algorithm: Random Forest Regressor
- Trees: 50
- Maximum depth: 6
- Minimum samples per leaf: 10
- Features: {len(saved_features)}

## Held-Out Test Performance
- MAE: {mean_absolute_error(y_test, test_predictions):.2f} m/h
- RMSE: {np.sqrt(mean_squared_error(y_test, test_predictions)):.2f} m/h
- R2: {r2_score(y_test, test_predictions):.4f}
- Baseline MAE: {mean_absolute_error(y_test, baseline_test_predictions):.2f} m/h

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
"""

summary_path = (
    project_root / "reports" / "model_evaluation_summary.md"
)

summary_path.write_text(
    evaluation_summary,
    encoding="utf-8"
)

print("\nSTEP 5.8 — MODEL EVALUATION SUMMARY")
print("Report saved:", summary_path)
print("Report characters:", len(evaluation_summary))
