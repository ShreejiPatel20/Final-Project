import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)



def inspect_data(df):
    print("\n" + "=" * 65)
    print("1. DATASET OVERVIEW")
    print("=" * 65)
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst five rows:")
    print(df.head().to_string(index=False))

    print("\nData types and non-null counts:")
    df.info()

    print("\nSummary statistics:")
    print(df.describe(include="all").T)

    print("\nMissing values per column:")
    print(df.isnull().sum().sort_values(ascending=False))

def clean_data(df):
   
    data = df.copy()

    original_rows = len(data)
    data = data.drop_duplicates().copy()
    print("\n" + "=" * 65)
    print("2. DATA CLEANING")
    print("=" * 65)
    print(f"Exact duplicate rows removed: {original_rows - len(data)}")

    for column in ["age", "fare"]:
        if column in data.columns and data[column].isnull().any():
            median_value = data[column].median()
            data[column] = data[column].fillna(median_value)
            print(f"Missing values in '{column}' filled with median: {median_value:.2f}")

    if "embarked" in data.columns and data["embarked"].isnull().any():
        modes = data["embarked"].mode()
        if not modes.empty:
            data["embarked"] = data["embarked"].fillna(modes[0])
            print("Missing values in 'embarked' filled with the mode.")

    data["survival_status"] = data["survived"].map(
        {0: "Did not survive", 1: "Survived"}
    )

    if {"sibsp", "parch"}.issubset(data.columns):
        data["family_size"] = data["sibsp"] + data["parch"] + 1

    print("\nMissing values after cleaning:")
    print(data.isnull().sum().sort_values(ascending=False))
    return data

def overall_survival(data):
    print("\n" + "=" * 65)
    print("3. OVERALL SURVIVAL")
    print("=" * 65)

    counts = data["survived"].value_counts().reindex([0, 1], fill_value=0)
    rate = data["survived"].mean() * 100
    print(f"Overall survival rate: {rate:.2f}%")
    print(f"Passengers who did not survive: {counts.loc[0]}")
    print(f"Passengers who survived: {counts.loc[1]}")

    summary = pd.DataFrame({
        "Status": ["Did not survive", "Survived"],
        "Passengers": [int(counts.loc[0]), int(counts.loc[1])],
        "Percentage": [
            counts.loc[0] / len(data) * 100,
            counts.loc[1] / len(data) * 100,
        ],
    })
    print("\nSurvival summary:")
    print(summary.to_string(index=False))

    sns.countplot(
        data=data,
        x="survival_status",
        order=["Did not survive", "Survived"],
    )
    plt.title("Overall Titanic Survival")
    plt.xlabel("Survival outcome")
    plt.ylabel("Number of passengers")
    plt.tight_layout()
    plt.show()

def survival_by_sex(data):
    print("\n" + "=" * 65)
    print("4. SURVIVAL BY SEX")
    print("=" * 65)

    rates = data.groupby("sex", observed=False)["survived"].mean().mul(100)
    print("Survival rate by sex (%):")
    print(rates.round(2).to_string())

    sns.barplot(data=data, x="sex", y="survived", errorbar=None)
    plt.title("Survival Rate by Sex")
    plt.xlabel("Sex")
    plt.ylabel("Survival rate (proportion)")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

def survival_by_class(data):
    print("\n" + "=" * 65)
    print("5. SURVIVAL BY PASSENGER CLASS")
    print("=" * 65)

    rates = data.groupby("pclass", observed=False)["survived"].mean().mul(100)
    print("Survival rate by passenger class (%):")
    print(rates.round(2).to_string())

    sns.barplot(
        data=data,
        x="pclass",
        y="survived",
        order=[1, 2, 3],
        errorbar=None,
    )
    plt.title("Survival Rate by Passenger Class")
    plt.xlabel("Passenger class (1 = first, 3 = third)")
    plt.ylabel("Survival rate (proportion)")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

def survival_by_age(data):
    print("\n" + "=" * 65)
    print("6. AGE DISTRIBUTION BY SURVIVAL")
    print("=" * 65)

    sns.histplot(
        data=data,
        x="age",
        hue="survival_status",
        bins=20,
        multiple="stack",
    )
    plt.title("Passenger Age Distribution by Survival Outcome")
    plt.xlabel("Age")
    plt.ylabel("Number of passengers")
    plt.tight_layout()
    plt.show()

def survival_by_sex_and_class(data):
    print("\n" + "=" * 65)
    print("7. SURVIVAL BY SEX AND PASSENGER CLASS")
    print("=" * 65)

    table = data.pivot_table(
        index="pclass",
        columns="sex",
        values="survived",
        aggfunc="mean",
    ).mul(100)
    print("Survival rate by class and sex (%):")
    print(table.round(2).to_string())

    sns.barplot(
        data=data,
        x="pclass",
        y="survived",
        hue="sex",
        order=[1, 2, 3],
        errorbar=None,
    )
    plt.title("Survival Rate by Sex and Passenger Class")
    plt.xlabel("Passenger class")
    plt.ylabel("Survival rate (proportion)")
    plt.ylim(0, 1)
    plt.legend(title="Sex")
    plt.tight_layout()
    plt.show()

def fare_analysis(data):
    print("\n" + "=" * 65)
    print("8. TICKET FARE AND SURVIVAL")
    print("=" * 65)

    print("Fare summary by survival outcome:")
    print(
        data.groupby("survival_status", observed=False)["fare"]
        .agg(["count", "median", "mean"])
        .round(2)
        .to_string()
    )

    sns.boxplot(data=data, x="survival_status", y="fare", showfliers=False)
    plt.title("Ticket Fare by Survival Outcome")
    plt.xlabel("Survival outcome")
    plt.ylabel("Fare")
    plt.tight_layout()
    plt.show()


def family_size_analysis(data):
    print("\n" + "=" * 65)
    print("9. FAMILY SIZE AND SURVIVAL")
    print("=" * 65)

    if "family_size" not in data.columns:
        print("Family-size columns are not available in this dataset.")
        return

    rates = data.groupby("family_size", observed=False)["survived"].mean().mul(100)
    print("Survival rate by family size (%):")
    print(rates.round(2).to_string())

    sns.barplot(data=data, x="family_size", y="survived", errorbar=None)
    plt.title("Survival Rate by Family Size")
    plt.xlabel("Family size (passenger included)")
    plt.ylabel("Survival rate (proportion)")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

def correlation_heatmap(data):
    print("\n" + "=" * 65)
    print("10. CORRELATION HEATMAP")
    print("=" * 65)

    numeric_data = data.select_dtypes(include="number")
    correlation = numeric_data.corr()

    plt.figure(figsize=(10, 7))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap of Numeric Variables")
    plt.tight_layout()
    plt.show()

def main():
    df = load_data()
    inspect_data(df)
    data = clean_data(df)

    overall_survival(data)
    survival_by_sex(data)
    survival_by_class(data)
    survival_by_age(data)
    survival_by_sex_and_class(data)
    fare_analysis(data)
    family_size_analysis(data)
    correlation_heatmap(data)

    print("\n" + "=" * 65)
    print("PROJECT COMPLETED")
    print("=" * 65)
    print(
        "Conclusion: Use the printed percentages and charts to describe how "
        "survival varies by sex, passenger class, age, fare, and family size. "
        "These are associations in the data and do not prove causation."
    )


if __name__ == "__main__":
    main()