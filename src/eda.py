
from pathlib import Path
import pandas as pd

# Locate the processed drilling dataset.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "usrop_processed.parquet"

# Load the verified dataset.
drilling_data = pd.read_parquet(DATA_FILE)

print("\nEXPLORATORY DATA ANALYSIS")
print("Dataset loaded successfully!")
print("Total records:", f"{len(drilling_data):,}")
print("Total columns:", len(drilling_data.columns))
print("Number of wells:", drilling_data["well_id"].nunique())


# Examine the distribution of our prediction target.
rop_column = "Rate of Penetration m/h"

rop_stats = drilling_data[rop_column].describe()

print("\nRATE OF PENETRATION (ROP) STATISTICS")
print(rop_stats.round(2).to_string())

# Check whether any ROP measurements are zero or negative.
zero_rop = (drilling_data[rop_column] == 0).sum()
negative_rop = (drilling_data[rop_column] < 0).sum()

print("\nZero ROP records:", f"{zero_rop:,}")
print("Negative ROP records:", f"{negative_rop:,}")


import matplotlib.pyplot as plt

# Plot the distribution of drilling penetration rates.
plt.figure(figsize=(10, 6))

plt.hist(
    drilling_data[rop_column],
    bins=50,
    color="steelblue",
    edgecolor="white",
    alpha=0.85
)

plt.axvline(
    drilling_data[rop_column].mean(),
    color="red",
    linestyle="--",
    label="Mean ROP"
)

plt.axvline(
    drilling_data[rop_column].median(),
    color="green",
    linestyle="--",
    label="Median ROP"
)

plt.title("Rate of Penetration Distribution — USROP Dataset")
plt.xlabel("Rate of Penetration (m/h)")
plt.ylabel("Number of Measurements")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.show()


# Compare drilling performance across individual wells.
well_rop_summary = drilling_data.groupby("well_id")[rop_column].agg(
    ["count", "mean", "median", "std", "min", "max"]
)

print("\nROP STATISTICS BY WELL")
print(well_rop_summary.round(2).to_string())


# Compare ROP distributions across the seven wells.
plt.figure(figsize=(11, 6))

well_rop_data = [
    drilling_data.loc[
        drilling_data["well_id"] == well_id, rop_column
    ]
    for well_id in sorted(drilling_data["well_id"].unique())
]

plt.boxplot(
    well_rop_data,
    tick_labels=[f"Well {i}" for i in range(7)],
    showfliers=False
)

plt.title("Rate of Penetration by Well — USROP Dataset")
plt.xlabel("Well")
plt.ylabel("Rate of Penetration (m/h)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.show()


# Examine the relationship between Weight on Bit and ROP.
wob_column = "Weight on Bit kkgf"

wob_rop_corr = drilling_data[
    [wob_column, rop_column]
].corr().iloc[0, 1]

print("\nWEIGHT ON BIT VS ROP")
print(f"Pearson correlation: {wob_rop_corr:.3f}")


# Sample records for a readable WOB-versus-ROP scatterplot.
plot_data = drilling_data.sample(
    n=10000,
    random_state=42
)

plt.figure(figsize=(10, 6))

plt.scatter(
    plot_data[wob_column],
    plot_data[rop_column],
    alpha=0.25,
    s=10,
    color="steelblue"
)

plt.title("Weight on Bit vs Rate of Penetration — USROP")
plt.xlabel("Weight on Bit (kkgf)")
plt.ylabel("Rate of Penetration (m/h)")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()


# Calculate WOB–ROP correlation separately for each well.
well_correlations = drilling_data.groupby("well_id").apply(
    lambda group: group[wob_column].corr(group[rop_column]),
    include_groups=False
)

print("\nWOB VS ROP CORRELATION BY WELL")

for well_id, correlation in well_correlations.items():
    print(f"Well {well_id}: {correlation:.3f}")


# Analyze the relationship between rotary speed and ROP.
rpm_column = "Average Rotary Speed rpm"

rpm_rop_corr = drilling_data[
    [rpm_column, rop_column]
].corr().iloc[0, 1]

print("\nROTARY SPEED VS ROP")
print(f"Overall Pearson correlation: {rpm_rop_corr:.3f}")

# Compare the relationship across individual wells.
rpm_well_corr = drilling_data.groupby("well_id").apply(
    lambda group: group[rpm_column].corr(group[rop_column]),
    include_groups=False
)

print("\nROTARY SPEED VS ROP CORRELATION BY WELL")

for well_id, correlation in rpm_well_corr.items():
    print(f"Well {well_id}: {correlation:.3f}")


# Analyze the relationship between surface torque and ROP.
torque_column = "Average Surface Torque kN.m"

torque_rop_corr = drilling_data[
    [torque_column, rop_column]
].corr().iloc[0, 1]

print("\nSURFACE TORQUE VS ROP")
print(f"Overall Pearson correlation: {torque_rop_corr:.3f}")

# Calculate the torque–ROP relationship separately for each well.
torque_well_corr = drilling_data.groupby("well_id").apply(
    lambda group: group[torque_column].corr(group[rop_column]),
    include_groups=False
)

print("\nSURFACE TORQUE VS ROP CORRELATION BY WELL")

for well_id, correlation in torque_well_corr.items():
    print(f"Well {well_id}: {correlation:.3f}")


# Analyze the relationship between standpipe pressure and ROP.
spp_column = "Average Standpipe Pressure kPa"

spp_rop_corr = drilling_data[
    [spp_column, rop_column]
].corr().iloc[0, 1]

print("\nSTANDPIPE PRESSURE VS ROP")
print(f"Overall Pearson correlation: {spp_rop_corr:.3f}")

# Compare pressure–ROP relationships across individual wells.
spp_well_corr = drilling_data.groupby("well_id").apply(
    lambda group: group[spp_column].corr(group[rop_column]),
    include_groups=False
)

print("\nSTANDPIPE PRESSURE VS ROP CORRELATION BY WELL")

for well_id, correlation in spp_well_corr.items():
    print(f"Well {well_id}: {correlation:.3f}")
    

# Analyze the relationship between mud flow rate and ROP.
flow_column = "Mud Flow In L/min"

flow_rop_corr = drilling_data[
    [flow_column, rop_column]
].corr().iloc[0, 1]

print("\nMUD FLOW RATE VS ROP")
print(f"Overall Pearson correlation: {flow_rop_corr:.3f}")

# Compare mud flow–ROP relationships across individual wells.
flow_well_corr = drilling_data.groupby("well_id").apply(
    lambda group: group[flow_column].corr(group[rop_column]),
    include_groups=False
)

print("\nMUD FLOW RATE VS ROP CORRELATION BY WELL")

for well_id, correlation in flow_well_corr.items():
    print(f"Well {well_id}: {correlation:.3f}")


# Analyze the relationship between gamma ray and ROP.
gamma_column = "USROP Gamma gAPI"

gamma_rop_corr = drilling_data[
    [gamma_column, rop_column]
].corr().iloc[0, 1]

print("\nGAMMA RAY VS ROP")
print(f"Overall Pearson correlation: {gamma_rop_corr:.3f}")

# Calculate gamma ray–ROP correlation for each well.
gamma_well_corr = drilling_data.groupby("well_id").apply(
    lambda group: group[gamma_column].corr(group[rop_column]),
    include_groups=False
)

print("\nGAMMA RAY VS ROP CORRELATION BY WELL")

for well_id, correlation in gamma_well_corr.items():
    print(f"Well {well_id}: {correlation:.3f}")


# Analyze the relationship between measured depth and ROP.
depth_column = "Measured Depth m"

depth_rop_corr = drilling_data[
    [depth_column, rop_column]
].corr().iloc[0, 1]

print("\nMEASURED DEPTH VS ROP")
print(f"Overall Pearson correlation: {depth_rop_corr:.3f}")

# Compare depth–ROP relationships within individual wells.
depth_well_corr = drilling_data.groupby("well_id").apply(
    lambda group: group[depth_column].corr(group[rop_column]),
    include_groups=False
)

print("\nMEASURED DEPTH VS ROP CORRELATION BY WELL")

for well_id, correlation in depth_well_corr.items():
    print(f"Well {well_id}: {correlation:.3f}")


# Examine the drilling profile of Well 3.
well_3 = drilling_data[
    drilling_data["well_id"] == 3
].sort_values("Measured Depth m")

plt.figure(figsize=(10, 7))

plt.plot(
    well_3[rop_column],
    well_3[depth_column],
    color="steelblue",
    linewidth=0.8,
    alpha=0.8
)

plt.gca().invert_yaxis()

plt.title("Well 3 — ROP Profile Along Measured Depth")
plt.xlabel("Rate of Penetration (m/h)")
plt.ylabel("Measured Depth (m)")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()


# Identify the highest ROP measurements in Well 3.
high_rop_records = well_3.nlargest(10, rop_column)

print("\nWELL 3 — TOP 10 ROP MEASUREMENTS")

print(
    high_rop_records[
        [
            depth_column,
            rop_column,
            wob_column,
            rpm_column
        ]
    ].round(2).to_string(index=False)
)


# Investigate unusually high ROP measurements in Well 3.
high_rop = well_3[well_3[rop_column] >= 95]

print("\nWELL 3 — HIGH ROP INVESTIGATION")
print("Records with ROP >= 95 m/h:", len(high_rop))
print("Unique ROP values:", high_rop[rop_column].nunique())

if not high_rop.empty:
    print(
        "Minimum measured depth:",
        round(high_rop[depth_column].min(), 2)
    )
    print(
        "Maximum measured depth:",
        round(high_rop[depth_column].max(), 2)
    )

    print("\nMost frequent high ROP values:")
    print(high_rop[rop_column].round(2).value_counts().head(10))


# Inspect high-ROP measurements without rounding their values.
high_rop_details = high_rop[
    [
        depth_column,
        rop_column,
        wob_column,
        rpm_column,
        torque_column,
        spp_column,
        flow_column
    ]
].sort_values(depth_column)

print("\nWELL 3 — HIGH ROP RECORD DETAILS")
print(high_rop_details.to_string(index=False))

print("\nExact unique ROP values:")
print(high_rop[rop_column].unique())


# Check consecutive repeated measurements in Well 3.
measurement_columns = [
    rop_column,
    wob_column,
    rpm_column,
    torque_column,
    spp_column,
    flow_column
]

# Sort measurements by depth before checking repetition.
well_3_sorted = well_3.sort_values(depth_column).reset_index(drop=True)

# Identify consecutive rows with identical operating measurements.
same_as_previous = (
    well_3_sorted[measurement_columns]
    .eq(well_3_sorted[measurement_columns].shift())
    .all(axis=1)
)

print("\nWELL 3 — REPEATED MEASUREMENT ANALYSIS")
print("Total records:", f"{len(well_3_sorted):,}")
print("Consecutive repeated records:", f"{same_as_previous.sum():,}")

repeat_percentage = same_as_previous.mean() * 100
print(f"Consecutive repetition rate: {repeat_percentage:.2f}%")


# Compare consecutive measurement repetition across all wells.
print("\nREPEATED MEASUREMENTS BY WELL")

for well_id, group in drilling_data.groupby("well_id"):
    group = group.sort_values(depth_column).reset_index(drop=True)

    repeated = (
        group[measurement_columns]
        .eq(group[measurement_columns].shift())
        .all(axis=1)
    )

    repeat_count = int(repeated.sum())
    repeat_pct = repeated.mean() * 100

    print(
        f"Well {well_id}: "
        f"{repeat_count:,} repeated records "
        f"({repeat_pct:.2f}%)"
    )


# Count distinct combinations of six drilling measurements.
print("\nUNIQUE DRILLING MEASUREMENT COMBINATIONS")

for well_id, group in drilling_data.groupby("well_id"):
    total_records = len(group)

    unique_records = group[
        measurement_columns
    ].drop_duplicates().shape[0]

    unique_pct = unique_records / total_records * 100

    print(
        f"Well {well_id}: "
        f"{unique_records:,} unique combinations "
        f"out of {total_records:,} records "
        f"({unique_pct:.2f}%)"
    )


# Examine ROP target diversity across wells.
print("\nROP TARGET DIVERSITY BY WELL")

for well_id, group in drilling_data.groupby("well_id"):
    total_records = len(group)
    unique_rop = group[rop_column].nunique()
    unique_pct = unique_rop / total_records * 100

    print(
        f"Well {well_id}: "
        f"{unique_rop:,} unique ROP values "
        f"out of {total_records:,} records "
        f"({unique_pct:.2f}%)"
    )


# Examine ROP target diversity across wells.
print("\nROP TARGET DIVERSITY BY WELL")

for well_id, group in drilling_data.groupby("well_id"):
    total_records = len(group)
    unique_rop = group[rop_column].nunique()
    unique_pct = unique_rop / total_records * 100

    print(
        f"Well {well_id}: "
        f"{unique_rop:,} unique ROP values "
        f"out of {total_records:,} records "
        f"({unique_pct:.2f}%)"
    )


# Examine the most frequently repeated ROP values in Well 2.
well_2 = drilling_data[drilling_data["well_id"] == 2]

rop_frequency = well_2[rop_column].value_counts()

print("\nWELL 2 — MOST FREQUENT ROP VALUES")
print(rop_frequency.head(10))

top_10_percentage = (
    rop_frequency.head(10).sum() / len(well_2) * 100
)

print(
    f"\nPercentage of Well 2 records represented "
    f"by the top 10 ROP values: {top_10_percentage:.2f}%"
)


# Identify consecutive runs of identical ROP values in Well 2.
well_2_sorted = well_2.sort_values(
    depth_column
).reset_index(drop=True)

# Assign a new group whenever the ROP value changes.
rop_run_id = (
    well_2_sorted[rop_column]
    .ne(well_2_sorted[rop_column].shift())
    .cumsum()
)

rop_runs = well_2_sorted.groupby(rop_run_id).agg(
    rop_value=(rop_column, "first"),
    record_count=(rop_column, "size"),
    start_depth=(depth_column, "min"),
    end_depth=(depth_column, "max")
)

longest_runs = rop_runs.nlargest(5, "record_count")

print("\nWELL 2 — LONGEST CONSECUTIVE ROP RUNS")
print(longest_runs.round(3).to_string(index=False))


# Examine operating conditions during the longest constant-ROP run.
longest_run_id = rop_runs["record_count"].idxmax()

longest_run_data = well_2_sorted[
    rop_run_id == longest_run_id
]

print("\nWELL 2 — LONGEST ROP RUN OPERATING CONDITIONS")

for column in [
    wob_column,
    rpm_column,
    torque_column,
    spp_column,
    flow_column
]:
    print(
        f"{column}: "
        f"{longest_run_data[column].nunique()} unique values"
    )


# Quantify operating parameter variation during the longest ROP run.
operating_columns = [
    wob_column,
    rpm_column,
    torque_column,
    spp_column,
    flow_column
]

print("\nWELL 2 — OPERATING PARAMETER VARIATION")

variation_summary = longest_run_data[
    operating_columns
].agg(["min", "max", "std"]).T

print(variation_summary.round(3).to_string())


# Rank numerical features by their correlation with ROP.
candidate_features = [
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

correlations = (
    drilling_data[candidate_features + [rop_column]]
    .corr()[rop_column]
    .drop(rop_column)
    .sort_values(key=abs, ascending=False)
)

print("\nROP — FEATURE CORRELATION RANKING")
print(correlations.round(3).to_string())
