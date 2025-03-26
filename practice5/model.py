import json
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Генерація фейкових історичних даних
np.random.seed(42)
num_samples = 100

temps = np.random.uniform(15, 35, num_samples)         # Температура (15–35°C)
humidities = np.random.uniform(30, 80, num_samples)    # Вологість повітря (30–80%)
soils = np.random.uniform(300, 800, num_samples)       # Вологість ґрунту (аналогове значення)

# Формула врожайності (із шумом)
yields = 0.05 * temps + 0.02 * humidities + 0.004 * soils + 0.8 + np.random.normal(0, 0.3, num_samples)

# Створення матриці X
X = np.vstack((temps, humidities, soils)).T
y = yields

# Навчання моделі
model = LinearRegression()
model.fit(X, y)

# Збереження моделі
joblib.dump(model, "./practice5/model.pkl")

# Збереження згенерованих даних у data.json
data = []
for t, h, s, yld in zip(temps, humidities, soils, yields):
    data.append({
        "temp": round(t, 2),
        "humidity": round(h, 2),
        "soil": round(s, 2),
        "yield": round(yld, 2)
    })

with open("./practice5/data.json", "w") as f:
    json.dump(data, f, indent=2)

# Виведення коефіцієнтів
coefficients = model.coef_
intercept = model.intercept_
coefficients, intercept