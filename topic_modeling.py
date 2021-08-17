import gensim
import nltk
import pandas as pd
from glob import glob
import csv
import re
from gensim import corpora
from gensim.corpora import MmCorpus
from gensim.test.utils import datapath
import warnings
import numpy as np
import os
from collections import defaultdict
new_model = gensim.models.ldamodel.LdaModel.load("../data/museum/model_topic/test_model.model")
nltk.download('punkt')
topic = []
stop_words = ['.','있다.','있는','이','의','를','에','을', '있', '하', '것', '들', '그', '되', 
              '수', '이', '보', '않', '없', '나', '사람', '주', '아니', '등', '같', '우리', '때', '년', 
              '가', '한', '지', '대하', '오', '말', '일', '그렇', '위하', '#', '⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀',
              '더', '다', '울', '앞에', '있다는', '대한', '것을', '나는', '앞에', '내', '✨', '하는',
             '큰', '그런', 'of', 'the', '있어요', '있다', '것이', '것이다', '_' , '등의', '및', '대해',
             '하고', '에서', '등이', '곳', '의해', '게', '있고', '이어', '⠀', '서', '인한', '⠀⠀⠀', '이번'
             , '°', '무', '몇', '두', '기자', '이를', '볼', '장', '자', '않고', '고', '따라', '곳에', '간'
             ,'위에', '둔']

                
with open('../data/museum/final.csv', encoding='UTF8') as f:
    rdr = csv.reader(f)
    dict_list = list()
    for line in rdr:
        if line[1] != 'text':
            caption_data = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', line[1])
            dict_list.append({'date':line[0],'caption':caption_data})
    caption = pd.DataFrame(dict_list)
    caption_concatenated = caption.groupby('date')['caption'].apply(''.join).reset_index()

    caption_concatenated['clean_caption'] = caption['caption'].replace("^[가-힣]", " ")
    #print(caption['clean_caption'])
    tokenized_caption = caption_concatenated['clean_caption'].apply(lambda x: x.split()) # 토큰화
#                 print(tokenized_caption)
    tokenized_caption = tokenized_caption.apply(lambda x: [item for item in x if item not in stop_words])
    dictionary1 = corpora.Dictionary(tokenized_caption)
    dictionary1.filter_extremes(no_below=2, no_above=0.5, keep_n=100000)
    dictionary1.compactify()
#                 print(dictionary1)
    corpus = [dictionary1.doc2bow(text_) for text_ in tokenized_caption]

    result = new_model[corpus]
    #caption_concatenated['lda_result'] = result

    d = defaultdict(lambda: defaultdict(int))

    for idx, rows in enumerate(result):
        for j,v in rows:
            d[j][idx] += v

    tuple_to_pandas=pd.DataFrame(d).fillna(0)



    result = pd.concat([caption_concatenated['date'],tuple_to_pandas],axis=1) 




    result.to_csv("../data/museum/final.csv")

for filename in files:
    
    print(filename)
    if '.csv' in filename:
        with open(os.path.join('../total_datas',filename), encoding='UTF8') as f:
            rdr = csv.reader(f)
            dict_list = list()
            for line in rdr:
                if line[1] != 'text':
                    caption_data = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', line[1])
                    dict_list.append({'date':line[0],'caption':caption_data})
            caption = pd.DataFrame(dict_list)
            caption_concatenated = caption.groupby('date')['caption'].apply(''.join).reset_index()
        
            caption_concatenated['clean_caption'] = caption['caption'].replace("^[가-힣]", " ")
            #print(caption['clean_caption'])
            tokenized_caption = caption_concatenated['clean_caption'].apply(lambda x: x.split()) # 토큰화
#                 print(tokenized_caption)
            tokenized_caption = tokenized_caption.apply(lambda x: [item for item in x if item not in stop_words])
            dictionary1 = corpora.Dictionary(tokenized_caption)
            dictionary1.filter_extremes(no_below=2, no_above=0.5, keep_n=100000)
            dictionary1.compactify()
#                 print(dictionary1)
            corpus = [dictionary1.doc2bow(text_) for text_ in tokenized_caption]
            
            result = new_model[corpus]
            #caption_concatenated['lda_result'] = result
            
            d = defaultdict(lambda: defaultdict(int))
            
            for idx, rows in enumerate(result):
                for j,v in rows:
                    d[j][idx] += v
            
            tuple_to_pandas=pd.DataFrame(d).fillna(0)
            
            
            
            result = pd.concat([caption_concatenated['date'],tuple_to_pandas],axis=1) 
            
            
        
            
            result.to_csv("./topic_modeling_"+filename)
            
              
                    
test.head()
