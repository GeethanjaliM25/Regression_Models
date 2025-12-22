# ...existing code...
import os
import sys
from pathlib import Path

# --- Dependency imports with helpful messages ---
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
except Exception:
    print("Missing core libraries. Install with:\n  py -3.13 -m pip install pandas numpy matplotlib")
    raise

# seaborn is optional
try:
    import seaborn as sns
except Exception:
    sns = None
    print("Optional: seaborn not found. Install with:\n  py -3.13 -m pip install seaborn")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.pipeline import make_pipeline
except Exception:
    print("Missing scikit-learn. Install with:\n  py -3.13 -m pip install scikit-learn")
    raise

# ----------------------------
# Configuration (absolute CSV path)
# ----------------------------
HERE = Path(__file__).resolve().parent
CSV_ABS = HERE / "stroke_data.csv" / "healthcare-dataset-stroke-data.csv"
CSV_NAME = CSV_ABS.name
TARGET_DEFAULT = "avg_glucose_level"

def read_csv_with_fallback(default_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(default_path)
    except PermissionError:
        abs_path = default_path.resolve()
        print(f"\n❌ Permission denied reading: {abs_path}")
        print("Possible fixes (PowerShell):")
        print(f"  dir \"{abs_path}\"")
        print(f"  icacls \"{abs_path}\"")
        print("If file is on OneDrive: right-click → 'Always keep on this device' or move file out of OneDrive.")
        print("If the file is open (Excel), close that program and try again.")
        alt = input("Enter alternate CSV path or press Enter to abort: ").strip()
        if not alt:
            print("Aborted.")
            sys.exit(1)
        alt_path = Path(alt)
        if not alt_path.exists():
            print(f"Alternate file not found: {alt_path}")
            sys.exit(1)
        return pd.read_csv(alt_path)
    except FileNotFoundError:
        print(f"\nFile not found: {default_path.resolve()}")
        alt = input("Enter full path to CSV (or press Enter to abort): ").strip()
        if not alt:
            print("Aborted.")
            sys.exit(1)
        alt_path = Path(alt)
        if not alt_path.exists():
            print(f"Alternate file not found: {alt_path}")
            sys.exit(1)
        return pd.read_csv(alt_path)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        sys.exit(1)

def main():
    # ----------------------------
    # Load dataset (attempt given absolute path first)
    # ----------------------------
    data = read_csv_with_fallback(CSV_ABS)
    print("✅ Dataset loaded:", CSV_NAME)
    print("Shape:", data.shape)
    print("Columns:", list(data.columns))

    # ----------------------------
    # Target selection
    # ----------------------------
    TARGET = TARGET_DEFAULT if TARGET_DEFAULT in data.columns else None
    if TARGET is None:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            print("No numeric columns available to use as target.")
            sys.exit(1)
        print(f"Default target '{TARGET_DEFAULT}' not found. Available numeric columns: {numeric_cols}")
        chosen = input(f"Enter target column name from above (or press Enter to use '{numeric_cols[0]}'): ").strip()
        TARGET = chosen if chosen else numeric_cols[0]
    print(f"Using target column: {TARGET}")

    # ----------------------------
    # Preprocessing
    # ----------------------------
    df = data.copy()

    for col in ["id", "Id", "ID"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Simple cleaning: strip whitespace from string columns
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()

    df = pd.get_dummies(df, drop_first=True)

    before = df.shape[0]
    df = df.dropna()
    after = df.shape[0]
    if after < before:
        print(f"Dropped {before - after} rows with missing values.")

    if TARGET not in df.columns:
        print(f"Target column '{TARGET}' missing after preprocessing.")
        sys.exit(1)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    if X.shape[1] == 0:
        print("No feature columns available after preprocessing.")
        sys.exit(1)

    # ----------------------------
    # Split and scale
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ----------------------------
    # Models
    # ----------------------------
    models = {
        "Linear Regression": make_pipeline(LinearRegression()),
        "Polynomial deg2 + LR": make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression()),
        "Ridge": make_pipeline(Ridge(alpha=1.0)),
        "Lasso": make_pipeline(Lasso(alpha=0.1, max_iter=10000)),
        "ElasticNet": make_pipeline(ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)),
        "Decision Tree": make_pipeline(DecisionTreeRegressor(random_state=42)),
        "Random Forest": make_pipeline(RandomForestRegressor(random_state=42, n_estimators=100)),
        "SVR": make_pipeline(SVR(kernel="rbf", C=1.0, epsilon=0.1))
    }

    results = {}

    for name, model in models.items():
        try:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results[name] = {"MSE": float(mse), "R2": float(r2), "Predictions": y_pred}
        except Exception as e:
            print(f"Model '{name}' failed: {e}")
            results[name] = {"MSE": None, "R2": None, "Predictions": None}

    # ----------------------------
    # Summary
    # ----------------------------
    print("\nModel performance:")
    for name, m in results.items():
        if m["R2"] is None:
            print(f"  {name}: failed")
        else:
            print(f"  {name}: R2={m['R2']:.4f}, MSE={m['MSE']:.4f}")

    # ----------------------------
    # Plots (safe)
    # ----------------------------
    plt.close("all")
    try:
        if sns is not None:
            sns.set(style="whitegrid")
        else:
            plt.style.use("seaborn-whitegrid")
    except Exception:
        pass

    try:
        corr = df.corr()
        plt.figure(figsize=(10, 6))
        if sns is not None:
            sns.heatmap(corr, cmap="coolwarm", square=True)
        else:
            plt.imshow(corr, cmap="coolwarm", aspect="auto")
            plt.colorbar()
            plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
            plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title("Correlation matrix")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Could not plot correlation matrix:", e)

    try:
        names = [n for n, v in results.items() if v["R2"] is not None]
        scores = [results[n]["R2"] for n in names]
        plt.figure(figsize=(10, 4))
        plt.bar(names, scores, color="skyblue")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("R2 score")
        plt.title("Model comparison")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Could not plot model comparison:", e)

    try:
        valid = [(n, v) for n, v in results.items() if v["Predictions"] is not None]
        if valid:
            cols = 2
            rows = (len(valid) + cols - 1) // cols
            plt.figure(figsize=(12, 4 * rows))
            for i, (name, metrics) in enumerate(valid):
                plt.subplot(rows, cols, i + 1)
                plt.scatter(y_test, metrics["Predictions"], s=10, alpha=0.7)
                mn = min(y_test.min(), np.min(metrics["Predictions"]))
                mx = max(y_test.max(), np.max(metrics["Predictions"]))
                plt.plot([mn, mx], [mn, mx], "--k", lw=0.8)
                plt.xlabel("Actual")
                plt.ylabel("Predicted")
                plt.title(name)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print("Could not plot Predicted vs Actual:", e)

    print("\n✅ Finished.")

if __name__ == "__main__":
    main()
# ...existing code...