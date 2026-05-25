from config import logger, timer_decorator, WC2026_QUALIFIERS_FILE, PREV_N, BASIC_FEATURES_FILE, FEATURES_LIST
import pandas as pd

def load_and_sort_data(path):
    df = pd.read_csv(path)                      # Грузим CSV в память
    df["date"] = pd.to_datetime(df["date"])     # Текст дат → машинный формат
    df = df.sort_values("date").reset_index(drop=True) # Сортируем хронологически, сбрасываем индексы
    return df                                   # Возвращаем готовый DataFrame

def calculate_result(df):
    df["result"] = 0                            # Создаём цель, по умолчанию 0 (ничья)
    df.loc[df["home_score"] > df["away_score"], "result"] = 1  # Счёт хозяев больше → 1 (П1)
    df.loc[df["home_score"] < df["away_score"], "result"] = 2  # Счёт гостей больше → 2 (П2)
    return df                                   # Возвращаем DataFrame с результатом

def points(f, a): return 3 if f > a else 1 if f == a else 0 # 3/1/0 за исход

def calculate_match_points(df):
    df["pts_H"] = [points(h, a) for h, a in zip(df["home_score"], df["away_score"])] # Очки хозяев
    df["pts_A"] = [points(a, h) for h, a in zip(df["home_score"], df["away_score"])] # Очки гостей
    return df                                   # Возвращаем DataFrame с очками

def calc_rolling(df, team_col, src_col, window, agg="sum"):
    def inner(grp):                             # Применяется к каждой команде отдельно
        s = grp.shift(1).rolling(window, min_periods=1) # Сдвиг на 1 матч + окно N игр
        return s.sum() if agg=="sum" else s.mean()      # Сумма (форма) или среднее (голы)
    return df.groupby(team_col)[src_col].transform(inner) # Возвращаем Series той же длины

def build_and_save(df, feature_cols, save_path):
    X = df[feature_cols]                        # Берём только признаки
    y = df["result"]                            # Берём только ответы
    final = pd.concat([X, y], axis=1).dropna()  # Склеиваем, удаляем строки без истории
    final.to_csv(save_path, index=False, encoding="utf-8") # Сохраняем без индексов
    print(f"Готово {len(final)} строк")         # Выводим итог

def add_market_balance(input_path, output_path):
    """
    >0.5 → хозяева дороже → бонус к П1.
    <0.5 → гости дороже → бонус к П2.
    0.5 → равны или нет данных → модель решит по другим признакам.
    """
    
    df = pd.read_csv(input_path)  # 1. Загружаем таблицу матчей
    total = df["home_market_value"] + df["away_market_value"]  # 2. Сумма стоимостей обеих команд
    # 3. Делим стоимость хозяев на сумму. Заменяем 0 на 1, чтобы не было ошибки деления
    df["bal_market_value"] = df["home_market_value"] / total.replace(0, 1)
    # 4. Если сумма была 0 (данных нет ни у кого), ставим нейтральный баланс 0.5
    df.loc[total == 0, "bal_market_value"] = 0.5
    # 5. Перемещаем столбец сразу после away_market_value для удобства (опционально)
    cols = df.columns.tolist()
    cols.insert(cols.index("away_market_value") + 1, cols.pop(cols.index("bal_market_value")))
    df = df[cols]
    df.to_csv(output_path, index=False, encoding="utf-8")  # 6. Сохраняем файл
    return df


@timer_decorator
def main():
    
    
    df = load_and_sort_data(f"{WC2026_QUALIFIERS_FILE}_clean.csv") # 1. Загрузка
    df = calculate_result(df)                   # 2. Цель
    df = calculate_match_points(df)             # 3. Очки
    df["prev_N_pts_H"] = calc_rolling(df, "home_team", "pts_H", PREV_N, "sum") # 4. Форма Х
    df["prev_N_pts_A"] = calc_rolling(df, "away_team", "pts_A", PREV_N, "sum") # 4. Форма Г
    # Для масштабирования просто добавляй новые вызовы calc_rolling:
    # df["prev_N_goals_H"] = calc_rolling(df, "home_team", "home_score", PREV_N, "mean")
    # col = ["home_elo_rating","away_elo_rating","home_elo_rank","away_elo_rank", "prev_N_pts_H","prev_N_pts_A"]
    build_and_save(df, FEATURES_LIST, f"{BASIC_FEATURES_FILE}.csv") # 5. Сборка и сохранение
    

    # add_market_balance("wc2026_qualifiers_clean.csv", "wc2026_qualifiers_clean.csv")
    
   
if __name__ == "__main__": main()