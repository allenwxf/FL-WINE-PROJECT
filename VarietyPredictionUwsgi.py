import os, json
import pandas as pd
import numpy as np
import keras
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request
varietyPrediction = Flask(__name__)


base_dir="/Users/wxf/Documents/GitHub/FL-WINE-PROJECT/"

data = pd.read_csv(base_dir + "dataset/wine-review/winemag-data-130k-v2-resampled.csv")
print(data.head(10))
description_train = data['description'][:]

encoder = LabelEncoder()
encoder.fit(data['variety'][:])

vocab_size = 12000  # 词袋数量
tokenize = keras.preprocessing.text.Tokenizer(num_words=vocab_size, char_level=False)
tokenize.fit_on_texts(description_train)

if os.path.exists(base_dir + "model/variety_prediction.h5"):
    combined_model = keras.models.load_model(base_dir + "model/variety_prediction.h5")
else:
    os.error("模型文件不存在")

@varietyPrediction.route('/')
def index():
    # request param
    desc = request.args.get("desc")
    price = request.args.get("price")

    description_bow_test = tokenize.texts_to_matrix(desc)
    test_embed = tokenize.texts_to_sequences(desc)

    predictions = combined_model.predict([description_bow_test, price] + [test_embed])
    if predictions.size > 0:
        prediction = predictions[0]
        index = np.argmax(prediction)
        variety_predicted = encoder.inverse_transform(index)
        return _ret(data={"variety_predicted":variety_predicted})


def _ret(msg="", errcode=0, data={}):
    ret = {
        "msg": msg,
        "code": errcode,
        "data": data
    }
    return json.dumps(ret)
