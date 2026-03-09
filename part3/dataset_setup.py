"""
Louis Han & Yu Chen
CSE 163 AG
This program implements data processing for the two datasets.
"""

import pandas as pd

saipe_df = pd.read_excel("est24all.xls", header=3)
ospi_df = pd.read_csv("Report_Card_Graduation_2024-25.csv", low_memory=False)


def merge_datasets(ospi_df: pd.DataFrame,
                   saipe_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the OSPI graduation data with the SAIPE poverty data
    at the county level, saves the merged dataset to a CSV file
    named 'merged_dataset.csv', and returns the merged DataFrame.
    """
    ospi_subset = ospi_df[
        (ospi_df["OrganizationLevel"] == "County") &
        (ospi_df["StudentGroup"] == "All Students") &
        (ospi_df["Cohort"] == "Seven Year")
    ].copy()

    ospi_subset["GraduationRate"] = pd.to_numeric(
        ospi_subset["GraduationRate"], errors="coerce"
    )
    ospi_final = ospi_subset[
        ["SchoolYear", "County", "Cohort", "GraduationRate"]
    ].copy()

    saipe_wa = saipe_df[saipe_df["Postal Code"] == "WA"].copy()
    saipe_wa["County_Key"] = saipe_wa["Name"].str.replace(
        " County", "", regex=False
    )
    saipe_wa["CountyFIPS"] = (
        saipe_wa["State FIPS Code"].astype(int).astype(str).str.zfill(2) +
        saipe_wa["County FIPS Code"].astype(int).astype(str).str.zfill(3)
    )

    saipe_cols = {
        "County_Key": "County",
        "CountyFIPS": "CountyFIPS",
        "Poverty Percent, All Ages": "PovertyPercent_AllAges",
        "Poverty Estimate, All Ages": "PovertyEstimate_AllAges",
        "Median Household Income": "MedianHouseholdIncome",
        "Poverty Percent, Age 0-17": "ChildPovertyPercent_0_17",
        "Poverty Estimate, Age 0-17": "ChildPovertyEstimate_0_17"
    }
    saipe_final = saipe_wa[list(saipe_cols.keys())].rename(columns=saipe_cols)

    merged_df = pd.merge(ospi_final, saipe_final, on="County", how="left")

    column_order = [
        "SchoolYear",
        "County",
        "CountyFIPS",
        "Cohort",
        "GraduationRate",
        "PovertyPercent_AllAges",
        "PovertyEstimate_AllAges",
        "MedianHouseholdIncome",
        "ChildPovertyPercent_0_17",
        "ChildPovertyEstimate_0_17"
    ]
    merged_df = merged_df[column_order]

    numeric_cols = [
        "GraduationRate",
        "PovertyPercent_AllAges",
        "PovertyEstimate_AllAges",
        "MedianHouseholdIncome",
        "ChildPovertyPercent_0_17",
        "ChildPovertyEstimate_0_17"
    ]
    for column in numeric_cols:
        merged_df[column] = pd.to_numeric(merged_df[column], errors="coerce")

    merged_df.to_csv("merged_dataset.csv", index=False)
    return merged_df


def main():
    print(ospi_df.isna().sum().sum())
    print(ospi_df.isna().sum().sort_values(ascending=False).head(10))

    print(ospi_df["GraduationRate"].describe())
    print(ospi_df["Cohort"].describe())
    print("--- Cohort Value Counts ---")
    print(ospi_df["Cohort"].value_counts())

    merged_df = merge_datasets(ospi_df, saipe_df)
    print(merged_df["GraduationRate"].describe())
    print(merged_df["PovertyPercent_AllAges"].describe())
    print(merged_df["MedianHouseholdIncome"].describe())
    print(merged_df["ChildPovertyPercent_0_17"].describe())


if __name__ == "__main__":
    main()
