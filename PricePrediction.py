import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder

from tensorflow import keras
layers = keras.layers

print("You have TensorFlow version", tf.__version__)

data = pd.read_csv("dataset/wine-review/winemag-data-130k-v2.csv")
data = data[:10000]
train_size = int(len(data) * 0.8)

# Train features
description_train = data['description'][:train_size]
variety_train = data['variety'][:train_size]
# Train labels
labels_train = data['price'][:train_size]

# Test features
description_test = data['description'][train_size:]
description_test.index = range(len(description_test))
variety_test = data['variety'][train_size:]
variety_test.index = range(len(variety_test))
# Test labels
labels_test = data['price'][train_size:]
labels_test.index = range(len(labels_test))


################
## Wide Model ##
################
# 1. 葡萄酒描述
vocab_size = 20000          # 词袋数量
tokenize = keras.preprocessing.text.Tokenizer(num_words=vocab_size, char_level=False)
tokenize.fit_on_texts(description_train)

description_bow_train = tokenize.texts_to_matrix(description_train)
description_bow_test = tokenize.texts_to_matrix(description_test)

# 2. 葡萄品种
# Use sklearn utility to convert label strings to numbered index
encoder = LabelEncoder()
encoder.fit(variety_train)
variety_train = encoder.transform(variety_train)
encoder = LabelEncoder()
encoder.fit(variety_test)
variety_test = encoder.transform(variety_test)
num_classes = np.max(variety_train) + 1
# Convert labels to one hot
variety_train = keras.utils.to_categorical(variety_train, num_classes)
variety_test = keras.utils.to_categorical(variety_test, num_classes)

# 搭建Wide模型
bow_inputs = layers.Input(shape=(vocab_size,))
variety_inputs = layers.Input(shape=(num_classes,))
merged_layer = layers.concatenate([bow_inputs, variety_inputs])
merged_layer = layers.Dense(256, activation='relu')(merged_layer)
predictions = layers.Dense(1)(merged_layer)
wide_model = keras.Model(inputs=[bow_inputs, variety_inputs], outputs=predictions)

wide_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])


################
## Deep Model ##
################
train_embed = tokenize.texts_to_sequences(description_train)
test_embed = tokenize.texts_to_sequences(description_test)

max_seq_length = 170
train_embed = keras.preprocessing.sequence.pad_sequences(train_embed, maxlen=max_seq_length)
test_embed = keras.preprocessing.sequence.pad_sequences(test_embed, maxlen=max_seq_length)

deep_inputs = layers.Input(shape=(max_seq_length,))
embedding = layers.Embedding(vocab_size, 8, input_length=max_seq_length)(deep_inputs)
embedding = layers.Flatten()(embedding)

embed_out = layers.Dense(1, activation='linear')(embedding)
deep_model = keras.Model(inputs=deep_inputs, outputs=embed_out)
deep_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])


################
##### MERGE ####
################
merged_out = layers.concatenate([wide_model.output, deep_model.output])
merged_out = layers.Dense(1)(merged_out)
combined_model = keras.Model(wide_model.input + [deep_model.input], merged_out)
combined_model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])

# Training
combined_model.fit([description_bow_train, variety_train] + [train_embed], labels_train, epochs=10, batch_size=128)

# Evaluation
combined_model.evaluate([description_bow_test, variety_test] + [test_embed], labels_test, batch_size=128)





predictions = combined_model.predict([description_bow_test, variety_test] + [test_embed])
print(predictions)


for i in range(15):
    val = predictions[i]
    print(description_test[i])
    print(val, 'Actual: ', labels_test.iloc[i], '\n')