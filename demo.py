import tweepy
import pandas as pd
import warnings
import re
import numpy as np
import math

# NTLK functions
import nltk
from nltk.corpus import stopwords
from nltk import tokenize as tok
from nltk.stem.snowball import SnowballStemmer # load nltk's SnowballStemmer as variabled 'stemmer'
import string
from nltk.tag import StanfordNERTagger
warnings.simplefilter("ignore", category=(FutureWarning, DeprecationWarning))
# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA
# Tf-Idf and Clustering packages
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from helper import *
from lda import *
from sen_analyze import get_sentiment_DL
from ner import get_NER


bearer_token = 'AAAAAAAAAAAAAAAAAAAAAFYukQEAAAAApZQpuKapMCSBFMHu%2Ba1bySvK2EM%3DJNec4foagBf1eRvl240UJxO8SnkXL6mwWkQXr80HKxA1JBCkoy'

def get_tweets(search_terms, max_results=100):
    tweet_fields=['id', 'author_id', 'text', 'lang','created_at', 'entities']
    complete_data = {}
    for term in search_terms:
        complete_data[term] = {}
        fields = 0
        print(term)
        query = f'{term} lang:en -is:retweet -has:media'
        for i,tweet_batch in  enumerate(tweepy.Paginator(client.search_recent_tweets, tweet_fields=tweet_fields,query=query, max_results=max_results, limit=100)):
            complete_data[term][fields] = tweet_batch
            fields+=1
    list_of_tweets = []
    for term in complete_data:
        for i in complete_data[term]:

            for tweet in complete_data[term][i].data:
                t_id= tweet.id
                author_id = tweet.author_id
                text = tweet.text
                date = tweet.created_at
                lang = tweet.lang
                temp = {'id':t_id, 'author_id':author_id,'text':text, 'date':date, "lang":lang, 'search_term':term }
                list_of_tweets.append(temp)
    tweet_df = pd.DataFrame(list_of_tweets)
    for _, tweet in tweet_df.iterrows(): 
        tweet_df['text'][_] = tweet.text.encode('ascii', 'ignore').decode('ascii') 
    tweet_df.to_csv("./data_with_search_term.csv")
    return tweet_df


if __name__ == "__main__":
    #Get words related to which we can search:
    search_terms = input("Enter words related to which you want analysis(space seperated:").lower().split()
    print(search_terms)
    client = tweepy.Client(bearer_token = bearer_token, wait_on_rate_limit=True)
    max_results = math.ceil(int(input("Enter the max no of tweets you want fetched for each word"))/100)
    tweet_df_all = get_tweets(search_terms, max_results)
    tweet_df_all = tweet_df_all[tweet_df_all['text'].notna()]
    print(tweet_df_all.shape)
    print(tweet_df_all.head())
    print(tweet_df_all.groupby('search_term')['id'].count())
    # remove urls and retweets and entities from the text
    tweet_df_all['text_clean'] = tweet_df_all['text'].apply(lambda row:clean_tweet(row))

    #remove punctuations
    RE_PUNCTUATION = '|'.join([re.escape(x) for x in string.punctuation])  
    tweet_df_all['text_clean'] = tweet_df_all['text_clean'].str.replace(RE_PUNCTUATION, "")
    print(tweet_df_all.head())
    # List of stopwords
    stop_words= stopwords.words('english') #import stopwords from NLTK package
    readInStopwords = pd.read_csv("./twitter-analytics/pre_process/twitterStopWords.csv", encoding='ISO-8859-1') # import stopwords from CSV file as pandas data frame
    readInStopwords = readInStopwords.wordList.tolist() # convert pandas data frame to a list
    readInStopwords.append('http')
    readInStopwords.append('https')

    # add in search terms as topic extraction is performed within each search topic, 
    # we do not want the word or valriation of the word captured as a topic word
    readInStopwords.extend(search_terms)
    stop_list = stop_words + readInStopwords # combine two lists i.e. NLTK stop words and CSV stopwords
    stop_list = list(set(stop_list)) # strore only unique values 
    print(tweet_df_all['search_term'].unique())
    tweets_all_topics = ldap(tweet_df_all, stop_list)
    tweets_all_topics = tweets_all_topics.reset_index(drop=True)
    print(tweets_all_topics.shape)
    print(tweets_all_topics.head())
    tweets_all_topics.to_csv('./demo/tweets_all_topics.csv', index=False)
    text_data =  tweets_all_topics
    text_out = get_sentiment_DL(text_data)
    print(text_out.sort_values(by='Sentiment_Score')[['text','Sentiment_Score']].head().T)
    print(text_out.sort_values(by='Sentiment_Score', ascending=False)[['text','Sentiment_Score']].head().T)
    text_out.to_csv('./demo/tweets_topics_sentiment.csv', index=False)
    text_ner_out = get_NER(text_out)
    #the outputs of the ner tagger
    print(text_ner_out.loc[(text_ner_out['Place'] != '') | (text_ner_out['Organization'] != '')|(text_ner_out['Person'] != '')][['text','Organization','Place','Person']].head())
    text_ner_out.to_csv('./processed_data/tweets_topics_sentiment_ner.csv', index=False)

