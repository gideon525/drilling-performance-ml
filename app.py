
import pandas as pd
from pathlib import Path
import pandas as pd

import streamlit as st
import joblib

# Configure the dashboard.
st.set_page_config(
    page_title="AI Drilling Performance Analytics",
    page_icon="🛢️",
    layout="wide"
)

# Main dashboard heading.
st.title("AI-Powered Drilling Performance Analytics")

st.markdown(
    """
    ### Rate of Penetration Prediction & Drilling Insights

    An interactive machine learning application for exploring
    drilling performance, predicting Rate of Penetration (ROP),
    and analyzing model performance across wells.

    **Dataset:** USROP — University of Stavanger

    **Machine Learning Model:** Random Forest Regressor
    """
)

st.divider()

# Sidebar navigation.
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Dashboard Section",
    [
        "Overview",
        "Drilling Data Explorer",
        "ROP Prediction",
        "Model Performance",
        "Feature Importance"
    ]
)

st.subheader(page)


# Load the processed drilling dataset.
project_root = Path(__file__).resolve().parent
data_path = project_root / "data" / "processed" / "usrop_processed.parquet"

@st.cache_data
def load_drilling_data():
    return pd.read_parquet(data_path)

drilling_data = load_drilling_data()

# Display dataset statistics on the Overview page.
if page == "Overview":
    st.subheader("Drilling Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Observations", f"{len(drilling_data):,}")
    col2.metric("Total Wells", drilling_data["well_id"].nunique())
    col3.metric(
        "Average ROP",
        f"{drilling_data['Rate of Penetration m/h'].mean():.2f} m/h"
    )

    
    # Summarize ROP statistics across all wells.
    st.subheader("ROP Performance by Well")

    well_summary = (
        drilling_data.groupby("well_id")
        .agg(
            Observations=("Rate of Penetration m/h", "size"),
            Mean_ROP=("Rate of Penetration m/h", "mean"),
            Median_ROP=("Rate of Penetration m/h", "median"),
            Min_ROP=("Rate of Penetration m/h", "min"),
            Max_ROP=("Rate of Penetration m/h", "max")
        )
        .round(2)
        .reset_index()
    )

    st.dataframe(
        well_summary,
        use_container_width=True,
        hide_index=True
    )

    
    # Compare average ROP across all seven wells.
    st.subheader("Average Rate of Penetration by Well")

    rop_chart_data = well_summary.copy()

    rop_chart_data["Well"] = (
        "Well " + rop_chart_data["well_id"].astype(str)
    )

    st.bar_chart(
        rop_chart_data,
        x="Well",
        y="Mean_ROP",
        x_label="Drilling Well",
        y_label="Average ROP (m/h)",
        color="#2474B5"
    )


# Interactive drilling data exploration.
if page == "Drilling Data Explorer":
    st.subheader("Explore Drilling Measurements")

    selected_well = st.selectbox(
        "Select a Well",
        sorted(drilling_data["well_id"].unique())
    )

    well_data = drilling_data[
        drilling_data["well_id"] == selected_well
    ]

    col1, col2, col3 = st.columns(3)

    col1.metric("Well", f"Well {selected_well}")
    col2.metric("Observations", f"{len(well_data):,}")
    col3.metric(
        "Average ROP",
        f"{well_data['Rate of Penetration m/h'].mean():.2f} m/h"
    )

    st.subheader("Drilling Measurements")

    st.dataframe(
        well_data.head(100),
        use_container_width=True
    )



    # Visualize ROP variation along the wellbore.
    st.subheader("Rate of Penetration vs. Measured Depth")

    rop_depth_data = (
        well_data[
            ["Measured Depth m", "Rate of Penetration m/h"]
        ]
        .sort_values("Measured Depth m")
        .set_index("Measured Depth m")
    )

    st.line_chart(
        rop_depth_data,
        y="Rate of Penetration m/h",
        x_label="Measured Depth (m)",
        y_label="Rate of Penetration (m/h)"
    )

    
    # Compare ROP with a selected drilling parameter.
    st.subheader("Drilling Parameter vs. ROP")

    parameter_options = [
        "Weight on Bit kkgf",
        "Average Rotary Speed rpm",
        "Average Surface Torque kN.m",
        "Average Standpipe Pressure kPa",
        "Mud Flow In L/min",
        "Average Hookload kkgf"
    ]

    selected_parameter = st.selectbox(
        "Select Drilling Parameter",
        parameter_options
    )

    comparison_data = well_data[
        [selected_parameter, "Rate of Penetration m/h"]
    ]

    st.scatter_chart(
        comparison_data,
        x=selected_parameter,
        y="Rate of Penetration m/h"
    )


# Display evaluation results from the frozen model.
if page == "Model Performance":
    st.subheader("Random Forest Model Performance")

    predictions_path = (
        project_root / "reports" / "final_test_predictions.csv"
    )

    @st.cache_data
    def load_test_predictions():
        return pd.read_csv(predictions_path)

    predictions_df = load_test_predictions()

    actual = predictions_df["Rate of Penetration m/h"]
    predicted = predictions_df["predicted_rop"]
    baseline = predictions_df["baseline_predicted_rop"]

    # Calculate evaluation metrics from saved predictions.
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score
    )
    import numpy as np

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    baseline_mae = mean_absolute_error(actual, baseline)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Model MAE", f"{mae:.2f} m/h")
    col2.metric("Model RMSE", f"{rmse:.2f} m/h")
    col3.metric("Model R²", f"{r2:.4f}")
    col4.metric("Baseline MAE", f"{baseline_mae:.2f} m/h")

    st.caption(
        "Evaluation on held-out Wells 5 and 6. "
        "The model was trained on Wells 0–4."
    )


    # Compare prediction errors for each held-out well.
    st.subheader("Random Forest vs. Baseline by Well")

    well_comparison = (
        predictions_df.groupby("well_id")
        .apply(
            lambda group: pd.Series({
                "Random Forest": mean_absolute_error(
                    group["Rate of Penetration m/h"],
                    group["predicted_rop"]
                ),
                "Baseline": mean_absolute_error(
                    group["Rate of Penetration m/h"],
                    group["baseline_predicted_rop"]
                )
            }),
            include_groups=False
        )
        .round(2)
    )

    well_comparison.index = [
        f"Well {well}" for well in well_comparison.index
    ]

    st.bar_chart(
        well_comparison,
        x_label="Held-Out Test Well",
        y_label="Mean Absolute Error (m/h)",
        stack=False
    )

    st.caption(
        "Lower MAE indicates better prediction accuracy. "
        "Well 5 and Well 6 were not used for model training."
    )

    
    # Display the saved actual-versus-predicted ROP plot.
    st.subheader("Actual vs. Predicted ROP")

    actual_predicted_path = (
        project_root
        / "reports"
        / "actual_vs_predicted_rop.png"
    )

    if actual_predicted_path.exists():
        st.image(
            str(actual_predicted_path),
            caption="Random Forest predictions on held-out Wells 5 and 6",
            width="stretch"
        )
    else:
        st.warning("Actual vs. Predicted ROP plot not found.")

    
    # Display residual errors for held-out wells.
    st.subheader("ROP Prediction Residual Analysis")

    residual_plot_path = (
        project_root
        / "reports"
        / "rop_prediction_residuals.png"
    )

    if residual_plot_path.exists():
        st.image(
            str(residual_plot_path),
            caption="Prediction error (Predicted ROP - Actual ROP)",
            width="stretch"
        )
    else:
        st.warning(
            "Residual plot not found. Check the saved filename."
        )


# Display feature importance from the trained Random Forest.
if page == "Feature Importance":
    st.subheader("Random Forest Feature Importance")

    importance_path = (
        project_root / "reports" / "rop_feature_importance.csv"
    )

    @st.cache_data
    def load_feature_importance():
        return pd.read_csv(importance_path)

    importance_df = load_feature_importance()

    st.dataframe(
        importance_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Feature Importance Ranking")

    chart_data = (
        importance_df
        .sort_values("importance_percent", ascending=False)
        .set_index("feature")
    )

    st.bar_chart(
        chart_data,
        y="importance_percent",
        horizontal=True,
        x_label="Importance (%)",
        y_label="Drilling Parameter"
    )

    st.caption(
        "Impurity-based Random Forest feature importance. "
        "These rankings describe model behavior, not causal effects."
    )


# Load the previously trained ROP prediction model.
if page == "ROP Prediction":
    st.subheader("Experimental ROP Prediction")

    st.warning(
        "Research prototype only. Predictions are not "
        "validated for operational drilling decisions."
    )

    model_path = (
        project_root / "models" / "rop_random_forest.joblib"
    )

    @st.cache_resource
    def load_rop_model():
        return joblib.load(model_path)

    rop_model = load_rop_model()

    st.success("Saved Random Forest model loaded successfully.")

    
    # Extract the trained estimator and its saved feature list.
    model_package = rop_model
    rop_model = model_package["model"]
    model_features = model_package["features"]

    st.write(
        type(rop_model).__name__
    )

    st.write(
        rop_model.n_features_in_
    )

    st.write(
        model_features
    )

    # Verify that the model and metadata agree.
    if rop_model.n_features_in_ == len(model_features) == 9:
        st.success("Model and feature metadata verified.")
    else:
        st.error("Model feature metadata mismatch.")
        st.stop()

    
    # Document the model's evaluation and data limitations.
    with st.expander("Model Performance & Important Limitations"):
        st.markdown("""
        **Model:** Random Forest Regressor

        **Training wells:** 0–4

        **Held-out test wells:** 5 and 6

        **Combined test results:**
        - Mean Absolute Error (MAE): 14.34 m/h
        - Root Mean Squared Error (RMSE): 18.23 m/h
        - R²: 0.1547
        - Baseline MAE: 17.70 m/h

        **Important limitations:**
        - Model accuracy varies substantially across wells.
        - The model overpredicts some low-ROP conditions
          and underpredicts some high-ROP conditions.
        - Repeated drilling measurements may affect
          model behavior and evaluation.
        - Mud density values for Well 3 were provisionally
          converted from suspected ppg to g/cm³.
          The original unit interpretation is unconfirmed.
        - Inputs within individual training ranges are
          not guaranteed to represent familiar conditions.
        - This is a contemporaneous ROP estimation prototype,
          not a validated advance forecasting system.
        - Bit wear or failure prediction has not been trained
          because labeled outcomes are unavailable.

        **Intended use:** Research, education, and portfolio
        demonstration only. Not for operational drilling decisions.
        """)

    
    # Build inputs using the model's saved feature order.
    st.subheader("Enter Drilling Parameters")

    training_data = drilling_data[
        drilling_data["well_id"].isin(
            model_package["training_wells"]
        )
    ]
    
    
    # Display-friendly labels; model feature names remain unchanged.
    feature_labels = {
        "Weight on Bit kkgf": "Weight on Bit (1,000 kgf)",
        "Average Standpipe Pressure kPa": "Standpipe Pressure (kPa)",
        "Average Surface Torque kN.m": "Surface Torque (kN·m)",
        "Average Rotary Speed rpm": "Rotary Speed (RPM)",
        "Mud Flow In L/min": "Mud Flow Rate (L/min)",
        "mud_density_corrected_g_cm3": "Mud Density (g/cm³)",
        "Diameter mm": "Hole Diameter (mm)",
        "Average Hookload kkgf": "Hookload (1,000 kgf)",
        "USROP Gamma gAPI": "Gamma Ray (API)"
    }


    input_values = {}

    with st.form("rop_prediction_form"):
        col1, col2, col3 = st.columns(3)

        for index, feature in enumerate(model_features):
            default_value = float(
                training_data[feature].median()
            )

            with [col1, col2, col3][index % 3]:
                input_values[feature] = st.number_input(
                    label=feature_labels.get(feature, feature),
                    value=default_value,
                    format="%.3f",
                    key=f"rop_input_{index}"
                )

        predict_button = st.form_submit_button(
            "Predict ROP"
        )

    
    if predict_button:
        # Preserve the feature names and order used during training.
        
        # Check inputs against ranges observed in training wells.
        out_of_range_features = []

        for feature in model_features:
            training_min = float(training_data[feature].min())
            training_max = float(training_data[feature].max())
            entered_value = input_values[feature]

            if not training_min <= entered_value <= training_max:
                out_of_range_features.append(
                    f"{feature}: {entered_value:.3f} "
                    f"(training range: {training_min:.3f} "
                    f"to {training_max:.3f})"
                )

        
        # Validate only after checking all nine features.
        if out_of_range_features:
            st.error(
                "Prediction blocked: drilling parameters "
                "outside the training-data ranges."
            )

            for message in out_of_range_features:
                st.write(f"- {message}")

            st.info(
                "Correct the highlighted values and try again."
            )

            st.stop()
        else:
            st.success(
                "All inputs are within the observed training-data ranges."
            )


        prediction_input = pd.DataFrame(
            [[input_values[feature] for feature in model_features]],
            columns=model_features
        )

        predicted_rop = float(
            rop_model.predict(prediction_input)[0]
        )

        # Calculate the difference from the dataset average.
        average_rop = float(
            drilling_data["Rate of Penetration m/h"].mean()
        )

        rop_difference = predicted_rop - average_rop

        st.subheader("Predicted Rate of Penetration")

        # Display prediction metrics side by side.
        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                label="Estimated ROP",
                value=f"{predicted_rop:.2f} m/h"
            )

        with result_col2:
            st.metric(
                label="Difference from Dataset Average",
                value=f"{rop_difference:+.2f} m/h",
                delta=f"{(rop_difference / average_rop) * 100:+.1f}%",
                delta_color="off"
            )
        # Explain the prediction's intended use.
        st.caption(
            "Estimated rate of penetration based on the entered drilling "
            "parameters. This is a research-stage model estimate, "
            "not an operational drilling recommendation."
        )

        # Compare predicted ROP with the dataset average.
        st.subheader("Predicted ROP vs. Dataset Average")

        rop_comparison = pd.DataFrame({
            "Category": [
                "Dataset Average",
                "Predicted ROP"
            ],
            "ROP (m/h)": [
                average_rop,
                predicted_rop
            ]
        })

        st.bar_chart(
            rop_comparison,
            x="Category",
            y="ROP (m/h)",
            horizontal=True
        )

        st.caption(
            "The dataset average is a descriptive reference, "
            "not a drilling performance target."
        )