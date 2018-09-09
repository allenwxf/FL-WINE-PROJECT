import grequests
import logging
import json
import pandas as pd
from googletrans import Translator
from googletrans.utils import format_json

headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
translator = Translator(service_urls=['translate.google.cn'])

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='log.txt')
logger = logging.getLogger()


def exception_handler(request, exception):
    logger.warning('exception when at %s :%s',request.url,exception)


def work(urls):
    reqs = (grequests.get(u["url"],verify=True, allow_redirects=True, timeout=4) for u in urls)
    res = grequests.map(reqs, exception_handler=exception_handler,size=200)
    linenums = (u["linenum"] for u in urls)
    return zip(linenums, res)

def totaltranslate(descs_file, translate_zh_file):

    descs = pd.read_csv(descs_file, header=None)

    initline = sum(1 for line in open(translate_zh_file))

    urls = []
    num = 0
    # for line in f:
    for i in range(len(descs)):
        if i < initline:
            continue

        line = descs.iloc[i][1]
        linenum = str(descs.iloc[i][0])
        print(i, line)
        num+=1

        line = line.strip()
        token = translator.token_acquirer.do(line)
        url="https://translate.google.cn/translate_a/single?client=t&sl=en&tl=zh&hl=de&dt=at&dt=bd&dt=ex&dt=" \
            "ld&dt=md&dt=qca&dt=rw&dt=rm&dt=ss&dt=t&ie=UTF-8&oe=UTF-8&otf=1&ssel=3&tsel=0&kc=1&tk={0}&q={1}"\
            .format(token,line)
        urls.append({"url":url, "linenum":linenum})

        file2 = open(translate_zh_file, mode='a+', encoding='utf-8')

        if len(urls) >= 5:
            res = work(urls)
            for linenum, r in res:
                if hasattr(r,'status_code'):
                    if r.status_code == 200:
                        try:
                            a=format_json(r.text)
                            target = ''.join([d[0] if d[0] else '' for d in a[0]])
                            source = ''.join([d[1] if d[1] else '' for d in a[0]])
                        except Exception as e:
                            logger.error('when format:%s',e)
                            logger.error('%s\n%s',r.text)
                            source = ''
                            target = ''
                        if len(source) != 0 and len(target) != 0:
                            file2.write(linenum + ',' + target+'\n')
                            print("target", target)
                        else:
                            file2.write('\n')
                            print("target", "\\n")
                    else:
                            file2.write('\n')
                            print("target", "\\n")
            urls = []
            logger.info('finish 50 sentence, now at %s',num)

        file2.close()

def sentencetranslate(line):
    line = line.strip()
    text = translator.translate(line,src='en',dest='zh-cn').text
    return text

def completetranslate(descs_file, translate_zh_file, translate_zh_fixed_file):
    file1 = open(translate_zh_file, mode='r', encoding='utf-8')
    file2 = open(translate_zh_fixed_file, mode='a', encoding='utf-8')
    i = 1
    # with open('de.txt', mode='r', encoding='utf-8') as f:
    descs = pd.read_csv(descs_file)
    # for line in f:
    for i in range(len(descs)):
        line = descs.iloc[i][1]
        linenum = str(descs.iloc[i][0])
        t = file1.readline()
        if len(t) == 1:  # 'only \n'
            text = sentencetranslate(line)
            file2.write(linenum + ',' + text + '\n')
        else:
            file2.write(t)
        i += 1
        if i % 100 == 0:
            print(i)
    file1.close()
    file2.close()


if __name__ == "__main__":
    descs_file = "../dataset/wine-review/winemag-data-130k-v2-desc.csv"

    translate_zh_file = '../dataset/wine-review/winemag-data-130k-v2-desc-zh-v2.txt'
    translate_zh_fixed_file = '../dataset/wine-review/winemag-data-130k-v2-desc-zh-fixed-v2.txt'


    totaltranslate(descs_file, translate_zh_file)
    completetranslate(descs_file, translate_zh_file, translate_zh_fixed_file)