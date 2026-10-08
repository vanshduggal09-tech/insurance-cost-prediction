import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split


# ============================================================
# 1. PATHS
# ============================================================

folder = os.path.dirname(os.path.abspath(__file__))

graphs_folder = os.path.join(folder, "graphs")

# Create graphs folder if it doesn't exist
os.makedirs(graphs_folder, exist_ok=True)


# ============================================================
# 2. LOAD DATA AND MODEL
# ============================================================

df = pd.read_csv(
    os.path.join(folder, "insurance.csv")
)

model = joblib.load(
    os.path.join(folder, "insurance_model.pkl")
)


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

df["isfemale"] = df["sex"].map({
    "male": 0,
    "female": 1
})

df["issmoker"] = df["smoker"].map({
    "no": 0,
    "yes": 1
})

df["region_southeast"] = df["region"].map(
    lambda x: 1 if x == "southeast" else 0
)


# ============================================================
# 4. FEATURES
# ============================================================

features = [
    "age",
    "bmi",
    "isfemale",
    "issmoker",
    "children",
    "region_southeast"
]

X = df[features]
y = df["expenses"]


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 6. PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 7. EDA — EXPENSES DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df["expenses"],
    kde=True
)

plt.title("Distribution of Insurance Expenses")
plt.xlabel("Expenses")
plt.ylabel("Count")

plt.tight_layout()

path = os.path.join(
    graphs_folder,
    "01_expenses_distribution.png"
)

plt.savefig(
    path,
    dpi=300,
    bbox_inches="tight"
)

print("Saved:", path)

plt.show()
plt.close()


# ============================================================
# 8. EDA — AGE VS EXPENSES
# ============================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="age",
    y="expenses"
)

plt.title("Age vs Insurance Expenses")
plt.xlabel("Age")
plt.ylabel("Expenses")

plt.tight_layout()

path = os.path.join(
    graphs_folder,
    "02_age_vs_expenses.png"
)

plt.savefig(
    path,
    dpi=300,
    bbox_inches="tight"
)

print("Saved:", path)

plt.show()
plt.close()


# ============================================================
# 9. EDA — BMI VS EXPENSES
# ============================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="bmi",
    y="expenses"
)

plt.title("BMI vs Insurance Expenses")
plt.xlabel("BMI")
plt.ylabel("Expenses")

plt.tight_layout()

path = os.path.join(
    graphs_folder,
    "03_bmi_vs_expenses.png"
)

plt.savefig(
    path,
    dpi=300,
    bbox_inches="tight"
)

print("Saved:", path)

plt.show()
plt.close()


# ============================================================
# 10. EDA — SMOKER VS EXPENSES
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="smoker",
    y="expenses"
)

plt.title("Smoker vs Insurance Expenses")
plt.xlabel("Smoker")
plt.ylabel("Expenses")

plt.tight_layout()

path = os.path.join(
    graphs_folder,
    "04_smoker_vs_expenses.png"
)

plt.savefig(
    path,
    dpi=300,
    bbox_inches="tight"
)

print("Saved:", path)

plt.show()
plt.close()


# ============================================================
# 11. EDA — CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(10, 7))

correlation = df.corr(
    numeric_only=True
)

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

path = os.path.join(
    graphs_folder,
    "05_correlation_heatmap.png"
)

plt.savefig(
    path,
    dpi=300,
    bbox_inches="tight"
)

print("Saved:", path)

plt.show()
plt.close()


# ============================================================
# 12. ML — ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(8, 6))

sns.scatterplot(
    x=y_test,
    y=y_pred
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.title("Actual vs Predicted Insurance Expenses")
plt.xlabel("Actual Expenses")
plt.ylabel("Predicted Expenses")

plt.tight_layout()

path = os.path.join(
    graphs_folder,
    "06_actual_vs_predicted.png"
)

plt.savefig(
    path,
    dpi=300,
    bbox_inches="tight"
)

print("Saved:", path)

plt.show()
plt.close()


# ============================================================
# 13. ML — RESIDUAL PLOT
# ============================================================

residuals = y_test - y_pred

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=y_pred,
    y=residuals
)

plt.axhline(
    0,
    linestyle="--"
)

plt.title("Residual Plot")
plt.xlabel("Predicted Expenses")
plt.ylabel("Residuals")

plt.tight_layout()

path = os.path.join(
    graphs_folder,
    "07_residual_plot.png"
)

plt.savefig(
    path,
    dpi=300,
    bbox_inches="tight"
)

print("Saved:", path)

plt.show()
plt.close()


# ============================================================
# 14. ML — RIDGE COEFFICIENTS
# ============================================================

coefficients = model.named_steps["ridge"].coef_

coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": coefficients
}).sort_values(
    "Coefficient"
)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=coef_df,
    x="Coefficient",
    y="Feature"
)

plt.title("Ridge Regression Feature Coefficients")
plt.xlabel("Coefficient")
plt.ylabel("Feature")

plt.tight_layout()

path = os.path.join(
    graphs_folder,
    "08_ridge_coefficients.png"
)

plt.savefig(
    path,
    dpi=300,
    bbox_inches="tight"
)

print("Saved:", path)

plt.show()
plt.close()


# ============================================================
# 15. FINISHED
# ============================================================

print("\n==========================================")
print("ALL GRAPHS SAVED SUCCESSFULLY!")
print("==========================================")
print(f"Location: {graphs_folder}")

print("\nFiles created:")

for file in sorted(os.listdir(graphs_folder)):
    print(" -", file)