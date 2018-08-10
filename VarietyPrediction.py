from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
layers = keras.layers

print("You have TensorFlow version", tf.__version__)

data = pd.read_csv("dataset/wine-review/winemag-data-130k-v2.csv")
data = data.sample(frac=1)
print(data.head())
print(data.shape)

data = data[pd.notnull(data['country'])]
data = data[pd.notnull(data['price'])]
data = data.drop(data.columns[0], axis=1)

variety_threshold = 500
value_counts = data['variety'].value_counts()
to_remove = value_counts[value_counts <= variety_threshold].index
data.replace(to_remove, np.nan, inplace=True)
data = data[pd.notnull(data['variety'])]

data = data[:10000]

train_size = int(len(data) * .8)
print("Train size: %d" % train_size)
print("Test size: %d" % (len(data) - train_size))

# Train features
description_train = data['description'][:train_size]

# Train labels
labels_train = data['variety'][:train_size]

# Test features
description_test = data['description'][train_size:]

# Test labels
labels_test = data['variety'][train_size:]


################
## Wide Model ##
################
# 1. 葡萄酒描述
vocab_size = 12000          # 词袋数量
tokenize = keras.preprocessing.text.Tokenizer(num_words=vocab_size, char_level=False)
tokenize.fit_on_texts(description_train)

description_bow_train = tokenize.texts_to_matrix(description_train)
description_bow_test = tokenize.texts_to_matrix(description_test)

# 2. 葡萄品种
# Use sklearn utility to convert label strings to numbered index
encoder = LabelEncoder()
encoder.fit(labels_train)
labels_train = encoder.transform(labels_train)
labels_test = encoder.transform(labels_test)
num_classes = np.max(labels_train) + 1

# # Convert labels to one hot
# labels_train = keras.utils.to_categorical(labels_train, num_classes)
# labels_test = keras.utils.to_categorical(labels_test, num_classes)
# print(labels_train[0:10])


# 搭建Wide模型
bow_inputs = layers.Input(shape=(vocab_size,))
# variety_inputs = layers.Input(shape=(num_classes,))
# merged_layer = layers.concatenate([bow_inputs, variety_inputs])
# merged_layer = layers.Dense(256, activation='relu')(merged_layer)
# predictions = layers.Dense(1)(merged_layer)
# wide_model = keras.Model(inputs=[bow_inputs, variety_inputs], outputs=predictions)
inter_layer = layers.Dense(256, activation='relu')(bow_inputs)
inter_layer = layers.Dense(num_classes)(inter_layer)
predictions = layers.Activation('softmax')(inter_layer)
wide_model = keras.Model(input=[bow_inputs], outputs=predictions)

print(wide_model.summary())

wide_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


################
## Deep Model ##
################
train_embed = tokenize.texts_to_sequences(description_train)
test_embed = tokenize.texts_to_sequences(description_test)

max_seq_length = 170
train_embed = keras.preprocessing.sequence.pad_sequences(train_embed, maxlen=max_seq_length, padding="post")
test_embed = keras.preprocessing.sequence.pad_sequences(test_embed, maxlen=max_seq_length, padding="post")

deep_inputs = layers.Input(shape=(max_seq_length,))
embedding = layers.Embedding(vocab_size, 8, input_length=max_seq_length)(deep_inputs)
embedding = layers.Flatten()(embedding)
# embed_out = layers.Dense(1)(embedding)
# deep_model = keras.Model(inputs=deep_inputs, outputs=embed_out)
embedding = layers.Dense(num_classes)(embedding)
embed_out = layers.Activation('softmax')(embedding)
deep_model = keras.Model(inputs=deep_inputs, outputs=embed_out)
print(deep_model.summary())

deep_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


################
##### MERGE ####
################
merged_out = layers.concatenate([wide_model.output, deep_model.output])
# merged_out = layers.Dense(1)(merged_out)
merged_out = layers.Dense(num_classes)(merged_out)
merged_out = layers.Activation('softmax')(merged_out)
combined_model = keras.Model(wide_model.input + [deep_model.input], merged_out)
print(combined_model.summary())

combined_model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

# Training
combined_model.fit([description_bow_train] + [train_embed], labels_train, epochs=10, batch_size=128)

# Evaluation
combined_model.evaluate([description_bow_test] + [test_embed], labels_test, batch_size=128)





predictions = combined_model.predict([description_bow_test] + [test_embed])
print(predictions)

num_predictions = 40
diff = 0

for i in range(num_predictions):
    val = predictions[i]
    print(description_test.iloc[i])
    print('Predicted: ', val[0], 'Actual: ', labels_test.iloc[i], '\n')
    diff += abs(val[0] - labels_test.iloc[i])