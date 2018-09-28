import os, json, requests
import pandas as pd
import numpy as np
import jieba
import keras
from sklearn.preprocessing import LabelEncoder
from keras import backend as K
from flask import Flask, request
app = Flask(__name__)

tf_serving_api = "http://192.168.0.106:8501/v1/models/VarietyPredictionZh:predict"
base_dir="/etc/docker-fl-wine-project/FL-WINE-PROJECT/"

jieba_dict_file = base_dir + "dataset/jieba-dict/dict.txt"
jieba.load_userdict(jieba_dict_file)

data = pd.read_csv(base_dir + "dataset/wine-review/winemag-data-130k-v2-zh-resampled.csv")
all_description = data['desc_zh_cut'][:]

encoder = LabelEncoder()
encoder.fit(data['variety'][:])

vocab_size = 12000  # 词袋数量
tokenize = keras.preprocessing.text.Tokenizer(num_words=vocab_size, char_level=False)
tokenize.fit_on_texts(all_description)

max_seq_length = 170



@app.route('/', methods=['GET', 'POST'])
def index():
    # if request.method == "POST":
    # request param
    desc = request.values.get("desc", "")
    desc = jieba_cut(desc)
    price = float(request.values.get("price", "0.0"))
    print(desc, price)

    desc = pd.Series([desc])
    price = pd.Series([price])

    description_bow_test = tokenize.texts_to_matrix(desc)
    test_embed = tokenize.texts_to_sequences(desc)
    test_embed = keras.preprocessing.sequence.pad_sequences(test_embed, maxlen=max_seq_length, padding="post")

    res = get_predicted(description_bow_test, price, test_embed)
    rindex = np.argmax(res)
    variety_predicted = encoder.inverse_transform(rindex)
    K.clear_session()
    return _ret(data={"variety_predicted": variety_predicted})

def get_predicted(description_bow_test, price, test_embed):
    payload = {
        "instances": [{"input_bow": description_bow_test.tolist(),
                       "input_price": price.tolist(),
                       "input_embed": test_embed.astype(np.float32).tolist()}]
    }

    r = requests.post(tf_serving_api, json=payload)
    rdict = json.loads(r.content.decode("utf-8"))
    return rdict["predictions"]

def jieba_cut(input):
    res = jieba.cut(input, cut_all=False, HMM=True)
    res = " ".join(res)
    return res

def _ret(msg="", errcode=0, data={}):
    ret = {
        "msg": msg,
        "code": errcode,
        "data": data
    }
    return json.dumps(ret)
