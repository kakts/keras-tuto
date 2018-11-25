from keras.models imoprt Sequential
from keras.layers import Dense, Activation

# コンストラクタにレイヤーのリストを渡す
# 最初のレイヤーに入力のshapeについての情報を与える必要がある


model = Sequential([
  Dense(32, input_shape=(784,)),
  Activation('relu'),
  Dense(10),
  Activation('softmax')
])

# compileメソッドでどのような学習を行うかを設定する
"""
# マルチクラス分類問題の場合
model.compile(optimizer = 'rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# binary分類問題の場合
model.compile(optimizer = 'rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 平均二乗誤差を最小化する回帰問題の場合
model.compile(optimizer = 'rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 訓練
# Kerasのモデルは Numpy配列として 入力データとラベルデータから訓練する
# モデルを訓練するときは 一般的にfit関数を使う


"""