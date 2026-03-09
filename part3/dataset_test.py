"""
Louis Han & Yu Chen
CSE 163 AG
This program implements tests for the merged dataset.
"""

import dataset_setup

df = dataset_setup.merge_datasets(
    dataset_setup.ospi_df, dataset_setup.saipe_df
)


def test_merged_dataset_size():
    """
    Checks the number of rows, columns, and unique counties.
    """
    assert df.shape == (39, 10)
    assert df["County"].nunique() == 39


def test_merged_dataset_cohort():
    """
    Checks that the Cohort column has only the expected value.
    """
    assert df["Cohort"].nunique() == 1
    assert "Seven Year" in df["Cohort"].unique()


def test_county_fips():
    """
    Checks that CountyFIPS exists and is properly formatted.
    """
    assert "CountyFIPS" in df.columns
    assert df["CountyFIPS"].notna().all()
    assert df["CountyFIPS"].astype(str).str.len().eq(5).all()


def test_merged_dataset_missing1():
    """
    Checks that the socio-economic columns have no missing values.
    """
    socio_cols = [
        "PovertyPercent_AllAges",
        "PovertyEstimate_AllAges",
        "MedianHouseholdIncome",
        "ChildPovertyPercent_0_17",
        "ChildPovertyEstimate_0_17",
    ]
    assert df[socio_cols].isna().sum().sum() == 0


def test_merged_dataset_missing2():
    """
    Checks that the GraduationRate column has at most 2 missing values.
    """
    missing_grad = df["GraduationRate"].isna().sum()
    assert 0 <= missing_grad <= 2


def test_merged_dataset_graduation_rate():
    """
    Checks that GraduationRate values are between 0 and 1 if not missing.
    """
    gr = df["GraduationRate"].dropna()
    assert ((gr >= 0) & (gr <= 1)).all()


def test_merged_dataset_poverty_income():
    """
    Checks that poverty percentages are between 0 and 100.
    """
    assert ((df["PovertyPercent_AllAges"] >= 0) &
            (df["PovertyPercent_AllAges"] <= 100)).all()
    assert ((df["ChildPovertyPercent_0_17"] >= 0) &
            (df["ChildPovertyPercent_0_17"] <= 100)).all()


def test_merged_dataset_estimates_income():
    """
    Checks that poverty estimates and median income are non-negative.
    """
    assert (df["PovertyEstimate_AllAges"] >= 0).all()
    assert (df["ChildPovertyEstimate_0_17"] >= 0).all()
    assert (df["MedianHouseholdIncome"] >= 0).all()


def main():
    test_merged_dataset_size()
    test_merged_dataset_cohort()
    test_county_fips()
    test_merged_dataset_missing1()
    test_merged_dataset_missing2()
    test_merged_dataset_graduation_rate()
    test_merged_dataset_poverty_income()
    test_merged_dataset_estimates_income()
    print("All tests passed!")


if __name__ == "__main__":
    main()
