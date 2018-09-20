import pandas as pd

desc_zh = pd.read_csv("../dataset/wine-review/winemag-data-130k-v2-desc-zh-fixed-v5.csv")
# desc_zh = pd.read_csv("/tmp/winemag-data-130k-v2-desc-zh-fixed-v5.csv")
init_pd = pd.read_csv("../dataset/wine-review/winemag-data-130k-v2-resampled.csv")

print(init_pd.columns)
print(init_pd[init_pd.columns[0]].head)

print(desc_zh.columns)
print(desc_zh[desc_zh.columns[0]].head)

merged_pd = init_pd.join(desc_zh, on="index", how="left", lsuffix='_left', rsuffix='_right')
print(merged_pd.head)
merged_pd.to_csv("../dataset/wine-review/winemag-data-130k-v2-zh-resampled.csv")