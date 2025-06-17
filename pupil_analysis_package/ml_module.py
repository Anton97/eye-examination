import os
import pandas as pd
from glob import glob
import numpy as np
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def load_labeled_csvs(base_folder):
    all_data = []
    label_map = {"norm": 0, "deviation": 1}
    for label_name, label in label_map.items():
        label_path = os.path.join(base_folder, label_name)
        csv_files = glob(os.path.join(label_path, "*.csv"))
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                df = df.dropna(subset=["Time", "Diameter of the left pupil (px)", "Diameter of the right pupil (px)"])
                df["label"] = label
                all_data.append(df)
            except Exception as e:
                print(f"Error in {file}: {e}")
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()

def augment_dataframe(df, n_augments=2, noise_std=0.05):
    augmented_dfs = []
    features = ["Time", "Diameter of the left pupil (px)", "Diameter of the right pupil (px)"]
    for _ in range(n_augments):
        noisy_df = df.copy()
        for col in features:
            noise = np.random.normal(0, noise_std * df[col].std(), size=len(df))
            noisy_df[col] += noise
        augmented_dfs.append(noisy_df)
    return pd.concat([df] + augmented_dfs, ignore_index=True)

def train_and_save_model(train_data_path, val_data_path, model_save_path="best_elasticnet_model.pkl"):
    df_train = load_labeled_csvs(train_data_path)
    df_val = load_labeled_csvs(val_data_path)

    if df_train.empty or df_val.empty:
        print("Not enough data to train the model.")
        return

    X_train = df_train[["Time", "Diameter of the left pupil (px)", "Diameter of the right pupil (px)"]]
    y_train = df_train["label"]

    X_val = df_val[["Time", "Diameter of the left pupil (px)", "Diameter of the right pupil (px)"]]
    y_val = df_val["label"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print("Selection of hyperparameters ElasticNetCV...")
    enet_cv = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                           alphas=np.logspace(-3, 1, 50),
                           cv=5,
                           random_state=42)
    enet_cv.fit(X_train_scaled, y_train)
    best_alpha = enet_cv.alpha_
    best_l1_ratio = enet_cv.l1_ratio_

    print(f"The best parameters: alpha = {best_alpha:.4f}, l1_ratio = {best_l1_ratio:.2f}")

    model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, random_state=42)

    best_r2 = -np.inf
    for epoch in range(1, 31):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
        r2 = r2_score(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)

        print(f"Epoch {epoch:02d} — MSE: {mse:.4f}, R²: {r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            joblib.dump(model, model_save_path)
            print(f"✅ The new best model is saved (R² = {best_r2:.4f})")

    print(f"📦 The final best model is saved as \'{model_save_path}\'")

def predict_with_model(video_results_df, model_path="best_elasticnet_model.pkl"):
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: The model was not found in the {model_path} path. Please train the model first.")
        return None

    X = video_results_df[["Time", "Diameter of the left pupil (px)", "Diameter of the right pupil (px)"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    predictions = model.predict(X_scaled)
    return predictions


