import pandas as pd

# 1. Загружаем файл в память
df = pd.read_csv("wc2026_qualifiers_clean.csv")

# 2. Удаляем столбец по имени
df = df.drop(columns=["bal_market_value"])  # pandas создаёт новую таблицу без этой колонки

# 3. Перезаписываем файл очищенной версией
df.to_csv("wc2026_qualifiers_clean.csv", index=False, encoding="utf-8")

print("✅ Столбец удалён, файл сохранён без него")