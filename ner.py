import nltk
from nltk.tag import StanfordNERTagger
def get_NER(text_data):
    #/Users/prajwalshreyas/Desktop/Singularity/dockerApps/ner-algo/stanford-ner-2015-01-30
    stanford_classifier = './twitter-analytics//models/ner/english.all.3class.distsim.crf.ser.gz'
    stanford_ner_path = './twitter-analytics//models/ner/stanford-ner.jar'

    #try:
        # Creating Tagger Object
    st = StanfordNERTagger(stanford_classifier, stanford_ner_path, encoding='utf-8')
    #except Exception as e:
    #       print (e)

    # Get keyword for the input data frame
    #keyword = tweetDataFrame.keyword.unique()
    # Subset column containing tweet text and convert to list
    # next insert a placeholder ' 12345678 ' to signify end of individual tweets

    print ('start get_NER')
    text_out = text_data.copy()
    doc = [ docs + ' 12345678 ' for docs in list(text_data['text'])]
    # ------------------------- Stanford Named Entity Recognition
    tokens = nltk.word_tokenize(str(doc))
    entities = st.tag(tokens) # actual tagging takes place using Stanford NER algorithm


    entities = [list(elem) for elem in entities] # Convert list of tuples to list of list
    print ('tag complete')
    for idx,element in enumerate(entities):
        try:
            if entities[idx][0] == '12345678':
                entities[idx][1] = "DOC_NUMBER"  #  Modify data by adding the tag "Doc_Number"
            #elif entities[idx][0].lower() == keyword:
            #    entities[idx][1] = "KEYWORD"
            # Combine First and Last name into a single word
            elif entities[idx][1] == "PERSON" and entities[idx + 1][1] == "PERSON":
                entities[idx + 1][0] = entities[idx][0] + '-' + entities[idx+1][0]
                entities[idx][1] = 'Combined'
            # Combine consecutive Organization names
            elif entities[idx][1] == 'ORGANIZATION' and entities[idx + 1][1] == 'ORGANIZATION':
                entities[idx + 1][0] = entities[idx][0] + '-' + entities[idx+1][0]
                entities[idx][1] = 'Combined'
        except IndexError:
            break
    print ('enumerate complete')
    # Filter list of list for the words we are interested in
    filter_list = ['DOC_NUMBER','PERSON','LOCATION','ORGANIZATION']
    entityWordList = [element for element in entities if any(i in element for i in filter_list)]

    entityString = ' '.join(str(word) for insideList in entityWordList for word in insideList) # convert list to string and concatenate it
    entitySubString = entityString.split("DOC_NUMBER") # split the string using the separator 'TWEET_NUMBER'
    del entitySubString[-1] # delete the extra blank row created in the previous step

    # Store the classified NERs in the main tweet data frame
    for idx,docNER in enumerate(entitySubString):
        docNER = docNER.strip().split() # split the string into word list
        # Filter for words tagged as Organization and store it in data frame
        text_out.loc[idx,'Organization'] =  ','.join([docNER[i-1]  for i,x in enumerate(docNER) if x == 'ORGANIZATION'])
        # Filter for words tagged as LOCATION and store it in data frame
        text_out.loc[idx,'Place'] = ','.join([docNER[i-1] for i,x in enumerate(docNER) if x == 'LOCATION'])
        # Filter for words tagged as PERSON and store it in data frame
        text_out.loc[idx,'Person'] = ','.join([docNER[i-1]  for i,x in enumerate(docNER) if x == 'PERSON'])

    print ('process complete')
    return text_out