import joblib, pandas as pd

MODEL_NAME = "wc2026_LogReg"

def predict(csv_path: str, models_dir: str = "models"):
 
    model, scaler, features = [joblib.load(f"models/{f}") for f in [f"{MODEL_NAME}_model.pkl", f"{MODEL_NAME}_scaler.pkl", f"{MODEL_NAME}_features.pkl"]]

    df = pd.read_csv(f"data/{csv_path}")
    # Финалы ЧМ = нейтральное поле
    if "is_neutral" in features: df["is_neutral"] = 1

    X = scaler.transform(df[features])  # Порядок столбцов гарантирован features
    
    preds = model.predict(X)
    probs = model.predict_proba(X)

    print(f"{'Дата':<12} | {'Матч':<39} | {'Прогноз':<24} | Вероятности")
    print("-" * 102)
    for i, row in df.iterrows():
        p0, p1, p2 = probs[i]
        cls = preds[i]
        max_p = max(p0, p1, p2)
        
        res_base = "Ничья" if cls == 0 else f"Победа: {row['home_team'] if cls==1 else row['away_team']}"
        icon = "++" if max_p >= 0.70 else ("+" if 0.50 <= max_p < 0.70 else "?")
        res = f"{icon} {res_base}"
        d = str(row['date']).split('-')
        date_fmt = f"{d[2]}.{d[1]}.{d[0][-2:]}"
        prob = f"П1 {p1:.1%}   Ничья {p0:.1%}   П2 {p2:.1%}"

        print(f"{date_fmt:<12} | {str(row['home_team'])[:18]:>18} - {str(row['away_team'])[:18]:<18} | {str(res)[:24]:<24} | {prob}")


predict("wc2026_final_clean.csv")