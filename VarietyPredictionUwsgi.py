import os, json
import pandas as pd
import numpy as np
import keras
from sklearn.preprocessing import LabelEncoder
from keras import backend as K
from flask import Flask, request
varietyPrediction = Flask(__name__)


base_dir="/Users/wxf/Documents/GitHub/FL-WINE-PROJECT/"

data = pd.read_csv(base_dir + "dataset/wine-review/winemag-data-130k-v2-resampled.csv")
all_description = data['description'][:]

encoder = LabelEncoder()
encoder.fit(data['variety'][:])

vocab_size = 12000  # 词袋数量
tokenize = keras.preprocessing.text.Tokenizer(num_words=vocab_size, char_level=False)
tokenize.fit_on_texts(all_description)

max_seq_length = 170



@varietyPrediction.route('/', methods=['GET', 'POST'])
def index():
    variety_predicted = ""
    # if request.method == "POST":
    # request param
    desc = request.values.get("desc", "")
    price = float(request.values.get("price", "0.0"))
    print(desc, price)

    desc = pd.Series([desc])
    price = pd.Series([price])

    description_bow_test = tokenize.texts_to_matrix(desc)
    test_embed = tokenize.texts_to_sequences(desc)
    test_embed = keras.preprocessing.sequence.pad_sequences(test_embed, maxlen=max_seq_length, padding="post")

    if os.path.exists(base_dir + "model/variety_prediction.h5"):
        combined_model = keras.models.load_model(base_dir + "model/variety_prediction.h5")
        predictions = combined_model.predict([description_bow_test, price] + [test_embed])
        if predictions.size > 0:
            prediction = predictions[0]
            index = np.argmax(prediction)
            variety_predicted = encoder.inverse_transform(index)
            K.clear_session()
            return _ret(data={"variety_predicted": variety_predicted})
    else:
        # os.error("模型文件不存在")
        return _ret("模型文件不存在", -1)
    # else:
    #     return _ret(data={"variety_predicted": variety_predicted})


def _ret(msg="", errcode=0, data={}):
    ret = {
        "msg": msg,
        "code": errcode,
        "data": data
    }
    return json.dumps(ret)
