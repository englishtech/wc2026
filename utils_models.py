# Импортируем нужные библиотеки
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import BASIC_FEATURES_FILE, FEATURES_LIST

# 1. Загружаем подготовленный файл с признаками
df = pd.read_csv(f"{BASIC_FEATURES_FILE}.csv")  # Читаем CSV в переменную df (DataFrame = таблица в памяти)

# 2. Выбираем 6 признаков, которые модель будет использовать для прогноза
FEATURES_LIST = FEATURES_LIST

# 3. Отделяем признаки (X) от целевой переменной (y)
X = df[FEATURES_LIST]  # X = матрица чисел (входы модели, то что мы знаем до матча)
y = df["result"]       # y = правильные ответы (0=ничья, 1=победа хозяев, 2=победа гостей)

# 4. Делим данные на обучающую и проверочную выборки
# test_size=0.2 → 20% данных уйдут на проверку, 80% на обучение
# random_state=42 → фиксирует случайность, чтобы при каждом запуске деление было одинаковым
# stratify=y → сохраняет пропорцию исходов (ничьих/побед) в обеих частях
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Масштабируем признаки, чтобы большие числа (Elo ~1500) не перебивали маленькие (баланс ~0.5)
sc = StandardScaler()                      # Создаём объект масштабирования
X_train_scaled = sc.fit_transform(X_train) # Запоминаем правила на тренировочных данных и сразу преобразуем
X_test_scaled = sc.transform(X_test)       # Преобразуем тестовые данные по тем же правилам (без подглядывания!)

# 6. Создаём и обучаем модель
model = LogisticRegression(max_iter=1000, random_state=42)  # max_iter=1000 даёт больше шагов для точной настройки
model.fit(X_train_scaled, y_train)        # Модель ищет математические связи между X и y

# 7. Проверяем точность на данных, которые модель не видела во время обучения
accuracy = model.score(X_test_scaled, y_test)  # Считаем долю правильных прогнозов
print(f"✅ Точность на тесте: {accuracy:.2%}") # Выводим результат в процентах

# 8. Выводим веса признаков для каждого из 3 исходов
# model.coef_ — это таблица размером (3 класса × 7 признаков)
# Каждая строка показывает, как признак влияет на вероятность конкретного исхода
"""
print("\n📊 Веса признаков (чем больше модуль числа, тем сильнее влияние):")
for class_name, weights in zip(["Ничья (0)", "Победа хозяев (1)", "Победа гостей (2)"], model.coef_):
    print(f"\n--- {class_name} ---")
    for feature_name, weight in zip(FEATURES_LIST, weights):
        print(f"  {feature_name}: {weight:+.4f}")  # +.4f покажет знак (+/-) и 4 знака после запятой
"""