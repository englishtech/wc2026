from config import logger, timer_decorator, WC2026_QUALIFIERS_FILE, PREV_N
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

FEATURES_LIST = [
        "home_elo_rating", 
        "away_elo_rating", 
        "home_elo_rank", 
        "away_elo_rank", 
        # "home_market_value", 
        # "away_market_value",
]
MODEL_NAME = "wc2026_LogReg"

def filter_qualifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Оставляет только матчи квалификации ЧМ (убирает товарняки, кубки и т.д.)."""
    # Фильтр по ключевым словам в названии турнира (регистронезависимый)
    mask = df["tournament"].str.contains("world cup qualifier", case=False, na=False)
    return df[mask].reset_index(drop=True)  # Сбрасываем индексы после фильтрации

def load_and_sort(path):
    """Загружает CSV, парсит даты и сортирует по хронологии."""
    df = pd.read_csv(path)                    # Загружаем сырые матчи
    df["date"] = pd.to_datetime(df["date"])   # Текст → машинный формат
    return df.sort_values("date").reset_index(drop=True) # Хронология + сброс индексов

def add_target(df):
    """Добавляет result: 1=победа Х, 2=победа Г, 0=ничья."""
    df["result"] = 0                          # По умолчанию ничья
    df.loc[df["home_score"] > df["away_score"], "result"] = 1 # Победа Х
    df.loc[df["home_score"] < df["away_score"], "result"] = 2 # Победа Г
    return df

# ===== Признаки =====
def add_form_features(df, n=PREV_N):
    """Добавляет форму команд (сумма очков за последние N матчей)"""
    pts = lambda f, a: 3 if f > a else 1 if f == a else 0 # Очки 3/1/0
    df["pts_H"] = [pts(h, a) for h, a in zip(df["home_score"], df["away_score"])]
    df["pts_A"] = [pts(a, h) for h, a in zip(df["home_score"], df["away_score"])]
    # shift(1) убирает текущий матч, rolling(n) берет N прошлых
    roll = lambda col, team: df.groupby(team)[col].transform(lambda x: x.shift(1).rolling(n, min_periods=1).sum())
    df["prev_N_pts_H"] = roll("pts_H", "home_team")
    df["prev_N_pts_A"] = roll("pts_A", "away_team")
    FEATURES_LIST.extend(["prev_N_pts_H", "prev_N_pts_A"])
    return df.drop(columns=["pts_H", "pts_A"]) # Чистим временные столбцы

def add_market_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет bal_market_value: доля стоимости хозяев в общей стоимости матча (0.0–1.0)."""
    total = df["home_market_value"] + df["away_market_value"]
    df["bal_market_value"] = df["home_market_value"] / total.replace(0, 1)
    df.loc[total == 0, "bal_market_value"] = 0.5
    FEATURES_LIST.append("bal_market_value")
    return df

def add_neutral_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет is_neutral: 1 если location != home_team, иначе 0."""
    df["is_neutral"] = (df["location"] != df["home_team"]).astype(int)
    FEATURES_LIST.append("is_neutral")
    return df

def add_goals_stats(df: pd.DataFrame, n: int = PREV_N) -> pd.DataFrame:
    """Добавляет средние забитые/пропущенные за последние N матчей."""
    roll = lambda col, team: df.groupby(team)[col].transform(
        lambda x: x.shift(1).rolling(n, min_periods=1).mean()
    )
    df["avg_goals_for_H"] = roll("home_score", "home_team")      # Голы хозяев в атаке
    df["avg_goals_against_H"] = roll("away_score", "home_team")  # Голы хозяев в защите
    df["avg_goals_for_A"] = roll("away_score", "away_team")
    df["avg_goals_against_A"] = roll("home_score", "away_team")
    FEATURES_LIST.extend(["avg_goals_for_H", "avg_goals_against_H", "avg_goals_for_A", "avg_goals_against_A"])

    df["goal_diff_H"] = df["avg_goals_for_H"] - df["avg_goals_against_H"]
    df["goal_diff_A"] = df["avg_goals_for_A"] - df["avg_goals_against_A"]
    FEATURES_LIST.extend(["goal_diff_H", "goal_diff_A"])
    
    return df

def add_clean_sheets(df: pd.DataFrame, n: int = PREV_N) -> pd.DataFrame:
    """Доля матчей без пропущенных голов за последние N игр."""
    def clean_rate(grp):
        # 1 если не пропустили, 0 если пропустили; shift(1) исключает текущий матч
        return grp.shift(1).rolling(n, min_periods=1).apply(lambda x: (x == 0).sum() / len(x))
    df["clean_sheet_H"] = df.groupby("home_team")["away_score"].transform(clean_rate)
    df["clean_sheet_A"] = df.groupby("away_team")["home_score"].transform(clean_rate)
    FEATURES_LIST.extend(["clean_sheet_H", "clean_sheet_A"])
    return df

# ===== Новые признаки =====
def add_new_feature(df: pd.DataFrame) -> pd.DataFrame:
    pass
# ===== Новые признаки =====

def build_xy(df):
    """
    Собирает матрицу признаков X и целевой вектор y, 
    оставляя только существующие столбцы и удаляя строки с пропусками.
    """
    valid = [c for c in FEATURES_LIST if c in df.columns] # Фильтруем существующие признаки
    clean = df[valid + ["result"]].dropna()               # Убираем строки без истории
    return clean[valid], clean["result"]                  # X = входы, y = ответы

# ===== Обучение модели =====
# @timer_decorator
def train_model(X: pd.DataFrame, y: pd.Series, test_size=0.0) -> LogisticRegression:
    """Разделяет данные, масштабирует, обучает модель и выводит метрики."""
    scaler = StandardScaler()
    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        print(f"Разделение: train = {len(X_train)}, test = {len(X_test)}")
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        model.fit(scaler.fit_transform(X_train), y_train)
        
        print(f"✅ Точность на тесте: {model.score(scaler.transform(X_test), y_test):.2%}")
    else:
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        model.fit(scaler.fit_transform(X), y)
        print("✅ Обучено на 100% данных")

    print(f"{'Признак':<25} {'Ничья (0)':>12} {'П1 (1)':>12} {'П2 (2)':>12}")
    for feat in FEATURES_LIST:
        idx = FEATURES_LIST.index(feat)
        print(f"{feat:<25} {model.coef_[0][idx]:+12.4f} {model.coef_[1][idx]:+12.4f} {model.coef_[2][idx]:+12.4f}")

    pipe = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(max_iter=1000))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
    print(f"CV: {scores.mean():.2%} ± {scores.std()*2:.2%}")

    return model, scaler

def save_model_artifacts(model, scaler, features_list: list, folder="models", name="wc2026"):
    """Сохраняет обученную модель и scaler в папку."""
    Path(folder).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, f"{folder}/{name}_model.pkl")
    joblib.dump(scaler, f"{folder}/{name}_scaler.pkl")
    joblib.dump(features_list, f"{folder}/{name}_features.pkl")
    print(f"✅ Артефакты сохранены в /{folder}")


def run_pipeline():
    df = load_and_sort(f"data/{WC2026_QUALIFIERS_FILE}_clean.csv") # 1. Загрузка
    # df = filter_qualifiers(df)       # Только матчи квалификации ЧМ
    # df = add_form_features(df)       # Сумма очков за последние N игр
    # df = add_market_balance(df)      # Баланс стоимости команд
    # df = add_goals_stats(df)         # Средние забитые/пропущенные за последние N игр
    # df = add_clean_sheets(df)        # Матчи без пропущенных голов за последние N игр

    df = add_target(df)              # Цель
    # df = add_neutral_flag(df)        # 1 если location != home_team, иначе 0

    print(df["result"].value_counts().sort_index())
        
    X, y = build_xy(df)              # 4. Сборка матрицы
    
    print(f"Матрица X: {X.shape}, Вектор y: {y.shape}")
    
    model, scaler = train_model(X, y, test_size=0.2) # test_size=0.0 для ВСЕХ матчей
    
    # save_model_artifacts(model, scaler, FEATURES_LIST, name=MODEL_NAME)
    

if __name__ == "__main__": 
    run_pipeline()