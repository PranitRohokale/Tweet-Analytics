import warnings
warnings.simplefilter("ignore", category=(FutureWarning, DeprecationWarning))
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from helper import tokenize_only, show_topics
from sklearn.decomposition import LatentDirichletAllocation as LDA
import numpy as np

number_topics = 5
number_words = 5
def ldap(tweet_df_comp, stop_list):
    tweets_all_topics= pd.DataFrame()
    # term frequency modelling
    for terms in tweet_df_comp['search_term'].unique():
        print(terms)
        tweets_search_topics  = tweet_df_comp[tweet_df_comp['search_term']==terms].reset_index(drop=True)
        corpus = tweets_search_topics['text_clean'].tolist()
        # print(corpus)
        tf_vectorizer = CountVectorizer(max_df=0.9, min_df=0.00, stop_words=stop_list, tokenizer=tokenize_only) # Use tf (raw term count) features for LDA.
        tf = tf_vectorizer.fit_transform(corpus)
        
        # Create and fit the LDA model
        model = LDA(n_components=number_topics, n_jobs=-1)
        id_topic = model.fit(tf)
        # Print the topics found by the LDA model
        print("Topics found via LDA:")
        topic_keywords = show_topics(vectorizer=tf_vectorizer, lda_model=model, n_words=number_words)        
        # Topic - Keywords Dataframe
        df_topic_keywords = pd.DataFrame(topic_keywords)
        df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
        df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
        df_topic_keywords = df_topic_keywords.reset_index()
        df_topic_keywords['topic_index'] = df_topic_keywords['index'].str.split(' ', n = 1, expand = True)[[1]].astype('int')
        print(df_topic_keywords)
        
        ############ get the dominat topic for each document in a data frame ###############
        # Create Document — Topic Matrix
        lda_output = model.transform(tf)
        # column names
        topicnames = ["Topic" + str(i) for i in range(model.n_components)]
        # index names
        docnames = ["Doc" + str(i) for i in range(len(corpus))]
        
        # Make the pandas dataframe
        df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)
        # Get dominant topic for each document
        dominant_topic = np.argmax(df_document_topic.values, axis=1)
        df_document_topic['dominant_topic'] = dominant_topic   
        df_document_topic = df_document_topic.reset_index()
            
        #combine all the search terms into one data frame
        tweets_topics = tweets_search_topics.merge(df_document_topic, left_index=True, right_index=True, how='left')
        tweets_topics_words = tweets_topics.merge(df_topic_keywords, how='left', left_on='dominant_topic', right_on='topic_index')
        tweets_all_topics = tweets_all_topics.append(tweets_topics_words)
    
    return tweets_all_topics