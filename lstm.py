import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt


file_path = 'C:/Users/user/Downloads/w14/w14/CTA_-_Ridership_-_Daily_Boarding_Totals.csv'

df = pd.read_csv(file_path, parse_dates=["service_date"])
df.columns = ["date", "day_type", "bus", "rail", "total"]
df = df.sort_values("date").set_index("date")
df = df.drop_duplicates()

data_train = df[["bus", "rail", "total"]]["2016-01":"2018-12"] / 1e6
data_valid = df[["bus", "rail", "total"]]["2019-01":"2019-05"] / 1e6
data_test = df[["bus", "rail", "total"]]["2019-06":] / 1e6

seq_length = 56  # 序列長度

# 創建時間序列數據集
train_ds = tf.keras.utils.timeseries_dataset_from_array(
    data_train.to_numpy(),
    targets=data_train[seq_length:],
    sequence_length=seq_length,
    batch_size=32,
    shuffle=True,
    seed=42
)
valid_ds = tf.keras.utils.timeseries_dataset_from_array(
    data_valid.to_numpy(),
    targets=data_valid[seq_length:],
    sequence_length=seq_length,
    batch_size=32
)
test_ds = tf.keras.utils.timeseries_dataset_from_array(
    data_test.to_numpy(),
    targets=data_test[seq_length:],
    sequence_length=seq_length,
    batch_size=32
)

# LSTM 模型
model = tf.keras.Sequential([
    LSTM(64, return_sequences=True, input_shape=[seq_length, 3]),
    LSTM(64, return_sequences=False),
    Dense(3)  # 3 表示多輸出（bus, rail, total）
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
# 訓練模型
history = model.fit(train_ds, validation_data=valid_ds, epochs=50)


# 查看模型結構
model.summary()




# 繪製訓練結果
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE', color='blue')
plt.plot(history.history['val_mae'], label='Validation MAE', color='orange')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.show()
# 預測與可視化
predict = model.predict(test_ds)
actual_values = data_test[seq_length:].to_numpy()
dates = data_test.index[seq_length:]

plt.figure(figsize=(12, 6))
plt.plot(dates, actual_values[:, 0], label='Actual Bus', color='blue', marker='o')
plt.plot(dates, predict[:, 0], label='Predicted Bus', color='red', linestyle='--', marker='x')

plt.plot(dates, actual_values[:, 1], label='Actual Rail', color='green', marker='o')
plt.plot(dates, predict[:, 1], label='Predicted Rail', color='orange', linestyle='--', marker='x')

plt.plot(dates, actual_values[:, 2], label='Actual Total', color='purple', marker='o')
plt.plot(dates, predict[:, 2], label='Predicted Total', color='brown', linestyle='--', marker='x')

plt.title('Actual vs Predicted (Bus, Rail, Total)')
plt.xlabel('Date')
plt.ylabel('Ridership (in millions)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
