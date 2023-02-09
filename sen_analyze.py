import keras
import h5py
from keras.models import model_from_json
from keras.models import load_model
import json
from nltk.tokenize import RegexpTokenizer
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter("ignore", category=(FutureWarning, DeprecationWarning))


weight_path = './twitter-analytics//models/dl_sentiment_model/best_weight_glove_bi_512.hdf5'
prd_model = load_model(weight_path)
word_idx = json.load(open("./twitter-analytics//models/dl_sentiment_model/word_idx.txt"))


def get_sentiment_DL(text_data):
    weight_path = './twitter-analytics//models/dl_sentiment_model/best_weight_glove_bi_512.hdf5'
    prd_model = load_model(weight_path)
    print(prd_model.summary())
    word_idx = json.load(open("./twitter-analytics//models/dl_sentiment_model/word_idx.txt"))

    live_list = []
    batchSize = len(text_data)
    live_list_np = np.zeros((56,batchSize))
    for index, row in text_data.iterrows():
        #print (index)
        text_data_sample = text_data['text'][index]
        # split the sentence into its words and remove any punctuations.
        tokenizer = RegexpTokenizer(r'\w+')
        text_data_list = tokenizer.tokenize(text_data_sample)

        labels = np.array(['1','2','3','4','5','6','7','8','9','10'], dtype = "int")
        #word_idx['I']
        # get index for the live stage
        data_index = np.array([word_idx[word.lower()] if word.lower() in word_idx else 0 for word in text_data_list])
        data_index_np = np.array(data_index)

        # padded with zeros of length 56 i.e maximum length
        padded_array = np.zeros(56)
        padded_array[:data_index_np.shape[0]] = data_index_np[:56]
        data_index_np_pad = padded_array.astype(int)


        live_list.append(data_index_np_pad)

    live_list_np = np.asarray(live_list)
    score = prd_model.predict(live_list_np, batch_size=batchSize, verbose=0)
    single_score = np.round(np.dot(score, labels)/10,decimals=2)

    score_all  = []
    for each_score in score:

        top_3_index = np.argsort(each_score)[-3:]
        top_3_scores = each_score[top_3_index]
        top_3_weights = top_3_scores/np.sum(top_3_scores)
        single_score_dot = np.round(np.dot(top_3_index, top_3_weights)/10, decimals = 2)
        score_all.append(single_score_dot)

    text_data['Sentiment_Score'] = pd.DataFrame(score_all)

    return text_data