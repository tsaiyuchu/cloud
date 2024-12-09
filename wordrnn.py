import pandas as np
tb=np.read_csv('https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/refs/heads/master/datasets/waimai_10k/waimai_10k.csv') #讀取資料
#print(tb.head()) #查閱資料內容

import re
pattern=re.compile('.{1}') #將字取出
first_sentence=pattern.findall(tb.review[0])
data=[pattern.findall(s) for s in tb.review]  #將每行字取出
#print(first_sentence)
#print(data)

#給予字標號
import tensorflow as tf
tokenizer= tf.keras.preprocessing.text.Tokenizer(filters='')
tokenizer.fit_on_texts(data)
number_data=tokenizer.index_word
#print(c)
#print(len(c))
#將每行的字轉為對應的標號 類似one-hot encoding
train_data= tokenizer.texts_to_sequences(data)
#print(d[0:3])
#每行長度不一樣所以要改成一樣的長度 用padding
train_data=tf.keras.preprocessing.sequence.pad_sequences(
    train_data,
    padding='post',
    truncating='post',
    maxlen=30
)
#print(train_data[0:3])

numwords=len(number_data)
#print(numwords)

embedding_dim=250

from tensorflow.keras import layers
embedding_layer=layers.Embedding(numwords+1, embedding_dim)
#print(embedding_layer(tf.constant([13,30,1])))

model=tf.keras.Sequential()
model.add(embedding_layer)
model.add(layers.SimpleRNN(64))
model.add(layers.Dense(2, activation='softmax'))


model.compile(optimizer='adam',
              loss=tf.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

model.summary()

model.fit(train_data, tb.label, epochs=10)

test=pattern.findall('超好吃')
test=[test]

test= tokenizer.texts_to_sequences(test)
test=tf.keras.preprocessing.sequence.pad_sequences(
    test,
    padding='post',
    truncating='post',
    maxlen=30
)
import numpy as pd
predict = model.predict(test)
print(f"Prediction probabilities: {predict}")  
predicted_class = pd.argmax(predict, axis=1)
print(f"Predicted class: {predicted_class}")   