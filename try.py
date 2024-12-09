import pandas as pd
import tensorflow as tf

# 讀取 CSV 檔案
file_path = 'D:/cloud/CTA_-_Ridership_-_Daily_Boarding_Totals.csv'

df = pd.read_csv(file_path)
print(df[0:5])

# 轉化為日期時間格式
df = pd.read_csv(file_path, parse_dates=["service_date"])

# 修改欄位名稱
df.columns = ["date", "day_type", "bus", "rail", "total"]

# 排序並設置索引
df = df.sort_values("date").set_index("date")

# 移除 total 欄位和重複行
df = df.drop("total", axis=1)  
df = df.drop_duplicates()

print("\n處理後的資料：\n", df.iloc[0:5])  # 所有資料
