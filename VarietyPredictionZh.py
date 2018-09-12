from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import jieba
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
layers = keras.layers

print("You have TensorFlow version", tf.__version__)

#初始化配置
prediction_model_file = "model/variety_prediction_zh.h5"
dataset_file = "dataset/wine-review/winemag-data-130k-v2-zh.csv"
dataset_resampled_file = "dataset/wine-review/winemag-data-130k-v2-zh-resampled.csv"
jieba_dict_file = "dataset/jieba-dict/dict.txt"

jieba.load_userdict(jieba_dict_file)

# resample the original data set or get the resampled according to the existense of model file
if not os.path.exists(prediction_model_file):
    # resample data set
    data = pd.read_csv(dataset_file)
    data = data.sample(frac=1)
    print(data.head())
    print(data.shape)

    data = data[pd.notnull(data['country'])]
    data = data[pd.notnull(data['price'])] 
    data = data.drop(data.columns[0], axis=1)

    variety_threshold = 100
    value_counts = data['variety'].value_counts()
    to_remove = value_counts[value_counts <= variety_threshold].index
    data.replace(to_remove, np.nan, inplace=True)
    data = data[pd.notnull(data['variety'])]

    data.to_csv(dataset_resampled_file)
else:
    # get resampled data set
    data = pd.read_csv(dataset_resampled_file)


# 中文分词
def jieba_cut(input):
    res = jieba.cut(input, cut_all=False, HMM=True)
    res = " ".join(res)
    print(res)
    return res
data['description_zh_cut'] = data.apply(lambda row: jieba_cut(row["description_zh"]), axis=1)


train_size = int(len(data) * .8)
print("Train size: %d" % train_size)
print("Test size: %d" % (len(data) - train_size))

# Train features
description_train = data['description_zh_cut'][:train_size]
price_train = data['price'][:train_size]
# Train labels
labels_train_arr = data['variety'][:train_size]
# Test features
description_test = data['description_zh_cut'][train_size:]
price_test = data['price'][train_size:]
# Test labels
labels_test_arr = data['variety'][train_size:]

all_varieties = data['variety'][:]
# if not os.path.exists(prediction_model_file):
#     all_varieties.to_csv("model/variety_labels.csv", header=['variety'])
all_description = data['description_zh_cut'][:]


encoder = LabelEncoder()
encoder.fit(all_varieties)
labels_train = encoder.transform(labels_train_arr)
labels_test = encoder.transform(labels_test_arr)
num_classes = np.max(labels_train) + 1
labels_train = keras.utils.to_categorical(labels_train, num_classes)
labels_test = keras.utils.to_categorical(labels_test, num_classes)

# WIDE 1. 葡萄酒描述
vocab_size = 12000          # 词袋数量
tokenize = keras.preprocessing.text.Tokenizer(num_words=vocab_size, char_level=False)
tokenize.fit_on_texts(all_description)

description_bow_train = tokenize.texts_to_matrix(description_train)
description_bow_test = tokenize.texts_to_matrix(description_test)

# WIDE 2. 葡萄酒价格
# Use sklearn utility to convert label strings to numbered index


# DEEP 1. 嵌入向量
train_embed = tokenize.texts_to_sequences(description_train)
test_embed = tokenize.texts_to_sequences(description_test)

max_seq_length = 170
train_embed = keras.preprocessing.sequence.pad_sequences(train_embed, maxlen=max_seq_length, padding="post")
test_embed = keras.preprocessing.sequence.pad_sequences(test_embed, maxlen=max_seq_length, padding="post")


if not os.path.exists(prediction_model_file):
    ################
    ## Wide Model ##
    ################
    # 搭建Wide模型
    bow_inputs = layers.Input(shape=(vocab_size,))
    price_inputs = layers.Input(shape=(1,))
    merged_layer = layers.concatenate([bow_inputs, price_inputs])
    merged_layer = layers.Dense(256, activation='relu')(merged_layer)
    inter_layer = layers.Dense(num_classes)(merged_layer)
    predictions = layers.Activation('softmax')(inter_layer)
    wide_model = keras.Model(inputs=[bow_inputs, price_inputs], outputs=predictions)

    print(wide_model.summary())

    wide_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    ################
    ## Deep Model ##
    ################
    deep_inputs = layers.Input(shape=(max_seq_length,))
    embedding = layers.Embedding(vocab_size, 8, input_length=max_seq_length)(deep_inputs)
    embedding = layers.Flatten()(embedding)
    embedding = layers.Dense(num_classes)(embedding)
    embed_out = layers.Activation('softmax')(embedding)
    deep_model = keras.Model(inputs=deep_inputs, outputs=embed_out)
    print(deep_model.summary())

    deep_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    ################
    ##### MERGE ####
    ################
    merged_out = layers.concatenate([wide_model.output, deep_model.output])
    merged_out = layers.Dense(num_classes)(merged_out)
    merged_out = layers.Activation('softmax')(merged_out)
    combined_model = keras.Model(wide_model.input + [deep_model.input], merged_out)
    print(combined_model.summary())

    combined_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Training
    combined_model.fit([description_bow_train, price_train] + [train_embed], labels_train, epochs=100, batch_size=128)

    # Evaluation
    score = combined_model.evaluate([description_bow_test, price_test] + [test_embed], labels_test, batch_size=128)
    print("%s: %.2f%%" % (combined_model.metrics_names[1], score[1] * 100))

    combined_model.save(prediction_model_file)
else:
    combined_model = keras.models.load_model(prediction_model_file)


# predict
predictions = combined_model.predict([description_bow_test, price_test] + [test_embed])

num_predictions = 60

for i in range(num_predictions):
    prediction = predictions[i]
    index = np.argmax(prediction)
    print(description_test.iloc[i])
    label_name = encoder.inverse_transform(index)

    print('index: ', index, 'label_name: ', label_name, ' Predicted: ', labels_train_arr.iloc[index],
          'Actual: ', labels_test_arr.iloc[i], '\n')
