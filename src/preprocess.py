
from pathlib import Path
import pandas as pd

# Locate the project folder regardless of where the script is run.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Keep original drilling data separate from processed data.
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

print("Project root:", PROJECT_ROOT)
print("Raw data folder:", RAW_DIR)
print("Processed data folder:", PROCESSED_DIR)


# Find only the seven published USROP benchmark files.
csv_files = sorted(RAW_DIR.glob("USROP_A*.csv"))

print("\nUSROP files found:", len(csv_files))

well_data = []

for well_id, file_path in enumerate(csv_files):
    df = pd.read_csv(file_path)

    # Preserve the source well for later validation and model splitting.
    df["well_id"] = well_id

    # Remove the CSV's exported index column.
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    well_data.append(df)

    print(f"Well {well_id}: {len(df):,} records")

# Combine all wells into one dataframe.
drilling_data = pd.concat(well_data, ignore_index=True)

print("\nCombined records:", f"{len(drilling_data):,}")
print("Total columns:", len(drilling_data.columns))


# Check for missing values across the combined dataset.
missing_values = drilling_data.isna().sum()

print("\nDATA QUALITY REPORT")
print("Missing values:", int(missing_values.sum()))

# Check physical constraints without modifying the measurements.
checks = {
    "Negative ROP": (drilling_data["Rate of Penetration m/h"] < 0).sum(),
    "Negative Weight on Bit": (drilling_data["Weight on Bit kkgf"] < 0).sum(),
    "Negative Rotary Speed": (drilling_data["Average Rotary Speed rpm"] < 0).sum(),
    "Non-positive Mud Density": (drilling_data["Mud Density In g/cm3"] <= 0).sum(),
    "Mud Density Above 3 g/cm3": (drilling_data["Mud Density In g/cm3"] > 3).sum(),
}

for check_name, count in checks.items():
    print(f"{check_name}: {count:,}")


# Preserve the original mud density measurements.
density_column = "Mud Density In g/cm3"
drilling_data["mud_density_original"] = drilling_data[density_column]

# Create a separate column for the provisional unit correction.
drilling_data["mud_density_corrected_g_cm3"] = drilling_data[density_column]

# Well 3 appears to contain values in pounds per gallon (ppg).
well_3_mask = drilling_data["well_id"] == 3
PPG_TO_G_CM3 = 0.119826427

drilling_data.loc[
    well_3_mask, "mud_density_corrected_g_cm3"
] *= PPG_TO_G_CM3

print("\nMUD DENSITY CORRECTION CHECK")
print("Well 3 records:", int(well_3_mask.sum()))
print("Original Well 3 density range:")
print(drilling_data.loc[well_3_mask, "mud_density_original"].agg(["min", "max"]))
print("Provisional corrected Well 3 density range:")
print(drilling_data.loc[well_3_mask, "mud_density_corrected_g_cm3"].agg(["min", "max"]))


# Compare original and provisionally corrected mud density by well.
density_summary = drilling_data.groupby("well_id").agg(
    original_min=("mud_density_original", "min"),
    original_max=("mud_density_original", "max"),
    corrected_min=("mud_density_corrected_g_cm3", "min"),
    corrected_max=("mud_density_corrected_g_cm3", "max"),
)

print("\nMUD DENSITY VALIDATION BY WELL")
print(density_summary.round(3).to_string())

# Confirm that only Well 3 was changed.
unchanged_wells = drilling_data["well_id"] != 3

assert (
    drilling_data.loc[unchanged_wells, "mud_density_original"]
    == drilling_data.loc[unchanged_wells, "mud_density_corrected_g_cm3"]
).all(), "Unexpected changes detected outside Well 3!"

print("\nSUCCESS: All other wells retain their original mud density values.")


# Create the processed data folder if necessary.
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Save all records, including original and provisional density columns.
output_file = PROCESSED_DIR / "usrop_processed.parquet"

drilling_data.to_parquet(output_file, index=False)

print("\nPROCESSED DATASET SAVED")
print("File:", output_file)
print("Records:", f"{len(drilling_data):,}")
print("Columns:", len(drilling_data.columns))

# Reload the saved file to verify its contents.
verified_data = pd.read_parquet(output_file)

# Confirm that saving did not change the dataset.
assert verified_data.shape == drilling_data.shape, "Dataset shape mismatch!"
assert verified_data.equals(drilling_data), "Saved data differs from the original!"

print("\nPARQUET VERIFICATION")
print("File reopened successfully.")
print("Records:", f"{len(verified_data):,}")
print("Columns:", len(verified_data.columns))
print("Data integrity: PASSED")
