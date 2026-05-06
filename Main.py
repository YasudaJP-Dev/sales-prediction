import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# =========================
# ① データ読み込み
# =========================
df = pd.read_csv(r'sales.csv')
df["Date"] = pd.to_datetime(df["Date"])

# 売上計算
df["Sales"] = df["Quantity"] * df["Price"]

# 日別売上に集約
daily_sales = df.groupby("Date")["Sales"].sum().reset_index()

# =========================
# ② 特徴量作成（ここが最重要）
# =========================

# 日付特徴量
daily_sales["Day"] = (daily_sales["Date"] - daily_sales["Date"].min()).dt.days
daily_sales["Weekday"] = daily_sales["Date"].dt.weekday
daily_sales["Month"] = daily_sales["Date"].dt.month

# ラグ特徴量
daily_sales["Lag1"] = daily_sales["Sales"].shift(1)
daily_sales["Lag2"] = daily_sales["Sales"].shift(2)

# 移動平均
daily_sales["Rolling3"] = daily_sales["Sales"].rolling(3).mean()

# 欠損削除
daily_sales = daily_sales.dropna()

# =========================
# ③ 学習データ作成
# =========================
features = ["Day", "Weekday", "Month", "Lag1", "Lag2", "Rolling3"]
X = daily_sales[features]
y = daily_sales["Sales"]

# 時系列なのでシャッフルしない
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

# =========================
# ④ モデル学習
# =========================
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# =========================
# ⑤ 予測＆評価
# =========================
pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
print("RMSE:", rmse)

# =========================
# ⑥ 可視化（重要）
# =========================
plt.figure()
plt.plot(y_test.values, label="Actual")
plt.plot(pred, label="Predicted")
plt.title("Sales Prediction")
plt.legend()
plt.show()

# =========================
# ⑦ 未来予測（簡易）
# =========================
last_row = daily_sales.iloc[-1].copy()

future_preds = []

for i in range(5):
    new_day = last_row["Day"] + 1

    new_row = {
        "Day": new_day,
        "Weekday": (last_row["Weekday"] + 1) % 7,
        "Month": last_row["Month"],
        "Lag1": last_row["Sales"],
        "Lag2": last_row["Lag1"],
        "Rolling3": (last_row["Sales"] + last_row["Lag1"] + last_row["Lag2"]) / 3
    }

    new_df = pd.DataFrame([new_row])
    pred_value = model.predict(new_df)[0]

    future_preds.append(pred_value)

    # 更新
    last_row["Day"] = new_row["Day"]
    last_row["Weekday"] = new_row["Weekday"]
    last_row["Lag2"] = new_row["Lag2"]
    last_row["Lag1"] = new_row["Lag1"]
    last_row["Sales"] = pred_value

print("\n未来予測（次の5日）")
for i, val in enumerate(future_preds):
    print(f"{i+1}日後: {val:.2f}")
