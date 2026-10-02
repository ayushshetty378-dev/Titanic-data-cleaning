import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

HERE = Path(__file__).parent
IN_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "train.csv"
OUT_CSV = HERE / "titanic_cleaned.csv"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)


def synthetic_titanic(n=891, seed=42):
    """Stand-in with the same schema/missingness pattern as Kaggle's train.csv.
    Only used if the real file is absent, so the pipeline can be tested."""
    rng = np.random.default_rng(seed)
    sex = rng.choice(["male", "female"], n, p=[0.65, 0.35])
    pclass = rng.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55])
    age = np.clip(rng.normal(30, 14, n), 0.5, 80).round(1)
    age[rng.random(n) < 0.20] = np.nan
    fare = np.round(rng.lognormal(3, 1, n) / (pclass * 0.8), 2)
    surv_p = 0.2 + 0.5 * (sex == "female") + 0.1 * (pclass == 1)
    cabin = np.where(rng.random(n) < 0.77, None, "C" + rng.integers(1, 150, n).astype(str))
    embarked = rng.choice(["S", "C", "Q"], n, p=[0.72, 0.19, 0.09]).astype(object)
    embarked[rng.choice(n, 2, replace=False)] = None
    return pd.DataFrame({
        "PassengerId": np.arange(1, n + 1),
        "Survived": (rng.random(n) < surv_p).astype(int),
        "Pclass": pclass,
        "Name": [f"Passenger, Mr. {i}" for i in range(n)],
        "Sex": sex, "Age": age,
        "SibSp": rng.integers(0, 4, n), "Parch": rng.integers(0, 3, n),
        "Ticket": [f"T{i}" for i in range(n)],
        "Fare": fare, "Cabin": cabin, "Embarked": embarked,
    })

if IN_PATH.exists():
    df = pd.read_csv(IN_PATH)
    print(f"Loaded {IN_PATH}")
else:
    print(f"!! {IN_PATH} not found - using SYNTHETIC data with the Titanic schema.")
    print("!! Download the real train.csv from Kaggle and re-run for real results.")
    df = synthetic_titanic()

print("\n=== First 10 rows ===")
print(df.head(10))
print("\n=== .info() ===")
df.info()
print("\n=== .describe() ===")
print(df.describe())
print("\n=== Missing values (before) ===")
missing_before = df.isnull().sum()
print(missing_before[missing_before > 0])

age_raw = df["Age"].copy()

df = df.drop(columns=["Cabin"])

df["Age"] = df["Age"].fillna(df["Age"].median())

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df["Fare"] = df["Fare"].fillna(df["Fare"].median())

df = df.drop(columns=["PassengerId", "Name", "Ticket"])

print("\n=== Missing values (after) ===")
print(df.isnull().sum().sum(), "missing values remain")

le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
print("\nSex encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

try:
    ohe = OneHotEncoder(sparse_output=False)   # sklearn >= 1.2
except TypeError:
    ohe = OneHotEncoder(sparse=False)
emb = ohe.fit_transform(df[["Embarked"]])
emb_cols = [f"Embarked_{c}" for c in ohe.categories_[0]]
df = pd.concat([df.drop(columns=["Embarked"]),
                pd.DataFrame(emb, columns=emb_cols, index=df.index).astype(int)], axis=1)


sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
sns.histplot(age_raw.dropna(), bins=30, kde=True, color="steelblue", ax=axes[0])
axes[0].set(title="Age distribution - before cleaning (NaNs excluded)", xlabel="Age")
sns.histplot(df["Age"], bins=30, kde=True, color="seagreen", ax=axes[1])
axes[1].set(title="Age distribution - after median imputation", xlabel="Age")
plt.tight_layout()
plt.savefig(FIG_DIR / "age_distribution.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 4.5))
sns.histplot(data=df, x="Age", hue="Survived", bins=30, kde=True,
             palette={0: "tomato", 1: "seagreen"}, multiple="layer")
plt.title("Age distribution by survival")
plt.tight_layout()
plt.savefig(FIG_DIR / "age_by_survival.png", dpi=150)
plt.close()

df.to_csv(OUT_CSV, index=False)
print(f"\nSaved cleaned dataset -> {OUT_CSV}  shape={df.shape}")
print("Columns:", list(df.columns))
print(df.head())