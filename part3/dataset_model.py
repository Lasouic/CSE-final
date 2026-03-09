"""
Louis Han & Yu Chen
CSE 163 AG
This program builds predictive models for county-level graduation rate.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import dataset_setup


def main():
    merged_df = dataset_setup.merge_datasets(dataset_setup.ospi_df,
                                             dataset_setup.saipe_df)

    columns = [
        "GraduationRate",
        "MedianHouseholdIncome",
        "PovertyPercent_AllAges",
        "PovertyEstimate_AllAges",
        "ChildPovertyPercent_0_17",
        "ChildPovertyEstimate_0_17"
    ]

    model_df = merged_df[columns].copy()

    for column in columns:
        model_df[column] = pd.to_numeric(model_df[column], errors="coerce")

    model_df = model_df.dropna()

    X = model_df[
        [
            "MedianHouseholdIncome",
            "PovertyPercent_AllAges",
            "PovertyEstimate_AllAges",
            "ChildPovertyPercent_0_17",
            "ChildPovertyEstimate_0_17"
        ]
    ]
    y = model_df["GraduationRate"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=163
    )

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    linear_preds = linear_model.predict(X_test)

    forest_model = RandomForestRegressor(
        n_estimators=100,
        random_state=163
    )
    forest_model.fit(X_train, y_train)
    forest_preds = forest_model.predict(X_test)

    print("Linear Regression Results")
    print("MSE:", mean_squared_error(y_test, linear_preds))
    print("R2:", r2_score(y_test, linear_preds))
    print()

    print("Random Forest Results")
    print("MSE:", mean_squared_error(y_test, forest_preds))
    print("R2:", r2_score(y_test, forest_preds))


if __name__ == "__main__":
    main()
