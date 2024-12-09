import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt


file_path = 'D:/cloud/CTA_-_Ridership_-_Daily_Boarding_Totals.csv'
df = pd.read_csv(file_path, parse_dates=["service_date"])
df.columns = ["date", "day_type", "bus", "rail", "total"]
df = df.sort_values("date").set_index("date")
df = df.drop("total", axis=1)  
df = df.drop_duplicates()  

# 數據分類
rail_train = df["rail"]["2016-01":"2018-12"]/ 1e6
rail_valid = df["rail"]["2019-01":"2019-05"]/ 1e6
rail_test = df["rail"]["2019-06":]/ 1e6

seq_length = 56  # 序列長度

# 創建時間序列數據集
train_ds = tf.keras.utils.timeseries_dataset_from_array(
    rail_train.to_numpy(),
    targets=rail_train[seq_length:],
    sequence_length=seq_length,
    batch_size=32,
    shuffle=True,
    seed=42
)
valid_ds = tf.keras.utils.timeseries_dataset_from_array(
    rail_valid.to_numpy(),
    targets=rail_valid[seq_length:],
    sequence_length=seq_length,
    batch_size=32
)

test_ds = tf.keras.utils.timeseries_dataset_from_array(
    rail_test.to_numpy(),
    targets=rail_test[seq_length:],
    sequence_length=seq_length,
    batch_size=32
)

# Deep RNN model
model = tf.keras.Sequential([
 tf.keras.layers.SimpleRNN(32, return_sequences=True, input_shape=[None, 1]),
 tf.keras.layers.SimpleRNN(32, return_sequences=True),
 tf.keras.layers.SimpleRNN(32),
 tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

#model.summary()

# 訓練
history=model.fit(train_ds, validation_data=valid_ds, epochs=20, batch_size=32)

#檢視訓練結果
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss',color='blue')
plt.plot(history.history['val_loss'], label='val Loss', color='orange')
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

#評估
predict = model.predict(test_ds)
actual_values = rail_test[seq_length:].to_numpy()
dates = rail_test.index[seq_length:]
plt.figure(figsize=(12, 6))
plt.plot(dates,actual_values, label='Actual', color='blue', marker='o')
plt.plot(dates,predict, label='Predictions', color='red', linestyle='--',marker='x')
plt.title('Actual vs Predictions')
plt.xlabel('Date')
plt.ylabel('ridership')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
