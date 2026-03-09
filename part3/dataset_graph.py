"""
Louis Han & Yu Chen
CSE 163 AG
This program implements visualizations for the two datasets.
"""

import json
import urllib.request
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import dataset_setup


def load_us_counties_geojson():
    """
    Loads the GeoJSON file for U.S. counties.
    """
    url = ("https://raw.githubusercontent.com/plotly/datasets/master/"
           "geojson-counties-fips.json")
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def plot_graduation_rate_distribution(subset):
    """
    Plots the distribution of graduation rates at the county level
    for all students.
    """
    plt.figure()
    sns.histplot(data=subset, x="GraduationRate", bins=30, kde=True)
    plt.title("Distribution of Graduation Rate (County Level, All Students)")
    plt.xlabel("Graduation Rate")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("graduation_rate_distribution.png")


def plot_graduation_rate_by_cohort(subset):
    """
    Plots the graduation rate by cohort using a boxplot.
    """
    order = ["Four Year", "Five Year", "Six Year", "Seven Year"]

    plt.figure()
    sns.boxplot(data=subset, x="Cohort", y="GraduationRate", order=order)
    plt.title("Graduation Rate by Cohort (County Level, All Students)")
    plt.xlabel("Cohort")
    plt.ylabel("Graduation Rate")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig("graduation_rate_by_cohort.png")


def plot_median_household_income_distribution(plot_df):
    """
    Plots the distribution of median household income.
    """
    plt.figure()
    sns.histplot(data=plot_df, x="MedianHouseholdIncome", bins=30, kde=True)
    plt.title("Distribution of Median Household Income")
    plt.xlabel("Median Household Income")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("median_household_income_distribution.png")


def plot_graduation_rate_vs_median_income(plot_df):
    """
    Plots a regression line of graduation rate versus median household income.
    """
    plt.figure()
    sns.regplot(data=plot_df, x="MedianHouseholdIncome", y="GraduationRate")
    plt.title("Graduation Rate vs. Median Household Income")
    plt.xlabel("Median Household Income")
    plt.ylabel("Graduation Rate")
    plt.tight_layout()
    plt.savefig("graduation_rate_vs_median_income.png")


def plot_graduation_rate_vs_median_income_joint(plot_df):
    """
    Plots a jointplot of graduation rate versus median household income.
    """
    joint = sns.jointplot(
        data=plot_df,
        x="MedianHouseholdIncome",
        y="GraduationRate",
        kind="scatter"
    )
    joint.figure.suptitle("Graduation Rate vs. Median Household Income",
                          y=1.02)
    joint.set_axis_labels("Median Household Income", "Graduation Rate")
    joint.figure.savefig("graduation_rate_vs_median_income_jointplot.png")


def plot_child_poverty_distribution(plot_df):
    """
    Plots the distribution of child poverty percentage.
    """
    plt.figure()
    sns.histplot(data=plot_df, x="ChildPovertyPercent_0_17", bins=20, kde=True)
    plt.title("Distribution of Child Poverty Percentage (Ages 0-17)")
    plt.xlabel("Child Poverty Percentage (Ages 0-17)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("child_poverty_distribution.png")


def plot_graduation_rate_vs_child_poverty(plot_df):
    """
    Plots a regression line of graduation rate versus child poverty percent.
    """
    plt.figure()
    sns.regplot(data=plot_df, x="ChildPovertyPercent_0_17",
                y="GraduationRate")
    plt.title("Graduation Rate vs. Child Poverty Percentage")
    plt.xlabel("Child Poverty Percentage (Ages 0-17)")
    plt.ylabel("Graduation Rate")
    plt.tight_layout()
    plt.savefig("graduation_rate_vs_child_poverty.png")


def plot_graduation_rate_map(plot_df):
    """
    Creates an interactive choropleth map of graduation rate by county.
    """
    counties_geojson = load_us_counties_geojson()

    fig = px.choropleth(
        plot_df,
        geojson=counties_geojson,
        locations="CountyFIPS",
        color="GraduationRate",
        color_continuous_scale="Blues",
        scope="usa",
        hover_name="County",
        hover_data={
            "CountyFIPS": False,
            "MedianHouseholdIncome": True,
            "ChildPovertyPercent_0_17": True
        },
        labels={"GraduationRate": "Graduation Rate"},
        title="Washington County Graduation Rates"
    )

    fig.update_geos(fitbounds="locations", visible=False)
    fig.write_html("graduation_rate_map.html")


def plot_income_map(plot_df):
    """
    Creates an interactive choropleth map of median household income by county.
    """
    counties_geojson = load_us_counties_geojson()

    fig = px.choropleth(
        plot_df,
        geojson=counties_geojson,
        locations="CountyFIPS",
        color="MedianHouseholdIncome",
        color_continuous_scale="Greens",
        scope="usa",
        hover_name="County",
        hover_data={
            "CountyFIPS": False,
            "GraduationRate": True,
            "ChildPovertyPercent_0_17": True
        },
        labels={"MedianHouseholdIncome": "Median Household Income"},
        title="Washington County Median Household Income"
    )

    fig.update_geos(fitbounds="locations", visible=False)
    fig.write_html("median_household_income_map.html")


def main():
    # OSPI dataset
    df = pd.read_csv("Report_Card_Graduation_2024-25.csv")
    subset = df.query("OrganizationLevel == 'County' "
                      "and StudentGroup == 'All Students'").copy()
    subset["GraduationRate"] = pd.to_numeric(subset["GraduationRate"],
                                             errors="coerce")
    subset = subset.dropna(subset=["GraduationRate", "Cohort"])

    plot_graduation_rate_distribution(subset)
    plot_graduation_rate_by_cohort(subset)

    # merged dataset
    merged_df = dataset_setup.merge_datasets(dataset_setup.ospi_df,
                                             dataset_setup.saipe_df)
    merged_df["GraduationRate"] = pd.to_numeric(merged_df["GraduationRate"],
                                                errors="coerce")
    merged_df["MedianHouseholdIncome"] = pd.to_numeric(
        merged_df["MedianHouseholdIncome"], errors="coerce")
    merged_df["ChildPovertyPercent_0_17"] = pd.to_numeric(
        merged_df["ChildPovertyPercent_0_17"], errors="coerce")

    plot_df = merged_df[
        [
            "County",
            "CountyFIPS",
            "GraduationRate",
            "MedianHouseholdIncome",
            "ChildPovertyPercent_0_17"
        ]
    ].dropna()

    plot_median_household_income_distribution(plot_df)
    plot_graduation_rate_vs_median_income(plot_df)
    plot_graduation_rate_vs_median_income_joint(plot_df)
    plot_child_poverty_distribution(plot_df)
    plot_graduation_rate_vs_child_poverty(plot_df)
    plot_graduation_rate_map(plot_df)
    plot_income_map(plot_df)


if __name__ == '__main__':
    main()
