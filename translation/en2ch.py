from google.cloud import translate
import pandas as pd

# client = translate.Client()
# print(client.get_languages())
client = translate.Client(target_language='zh')

def wine_translate(input):
    res = client.translate(input, source_language='en')
    return res["translatedText"]

data = pd.read_csv("../dataset/wine-review/winemag-data-130k-v2-resampled.csv")
somedf = data.head()
data['description_zh'] = data.apply(lambda row: wine_translate(row["description"]), axis=1)

data.to_csv("../dataset/wine-review/winemag-data-130k-v2-resampled-zh.csv")