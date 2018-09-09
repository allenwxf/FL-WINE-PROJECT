from google.cloud import translate
import pandas as pd
import time

# client = translate.Client()
# print(client.get_languages())
client = translate.Client(target_language='zh')

def wine_translate(input):
    res = client.translate(input)
    print(res)
    return res

data = pd.read_csv("../dataset/wine-review/winemag-data-130k-v2-resampled.csv")
data["description"].to_csv("../dataset/wine-review/winemag-data-130k-v2-desc.csv")
exit(0)
somedf = data.head()

# 量太大，不适用一条一条翻译
# data['description_zh'] = data.apply(lambda row: wine_translate(row["description"]), axis=1)

batchSize = 5
start = time.clock()
chars = 0

for i in range(len(data)):
    # if i < 99185:
    #     continue

    if i % batchSize == 0:
        print("======batch no." + str(i//batchSize) + "...\n")
        desclist = []
        for j in range((i // batchSize) * batchSize, (i // batchSize + 1) * batchSize):
            if j in data.index:
                print(j, data.iloc[j]["description"])
                desclist.append(data.iloc[j]["description"])
                chars = chars + len(data.iloc[j]["description"])

        print("======char len: " + str(chars) + "\n")

        # gcloud免费账号默认100000c/100s，自适应时间间隔调整
        current = time.clock()
        if current < 120 + start:
            if chars >= 90000:
                print("======sent chars >= 10,0000/100s, sleep " + str(120 - current + start) + "s and continue")
                time.sleep(120 - current + start)
                start = time.clock()
                chars = 0
        else:
            start = time.clock()
            chars = 0

        print("======batch no." + str(i//batchSize) + " start translating...\n")
        desc_translated = wine_translate(desclist)

        for j in range((i // batchSize) * batchSize, (i // batchSize + 1) * batchSize):
            if j in data.index:
                print(j, desc_translated[j % batchSize]["translatedText"])
                data.loc[j, "description_zh"] = desc_translated[j % batchSize]["translatedText"]

        data.to_csv("../dataset/wine-review/winemag-data-130k-v2-resampled-zh.csv")


