from keras.models import Sequential
from keras.layers import Dense, Activation

import numpy as np

# 1つの入力から２クラス分類をするモデル
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# ダミーデータ作成
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

# 各イテレーションのバッチサイズ32で学習を行う
model.fit(data, labels, epochs=300, batch_size=32)
