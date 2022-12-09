import re
import pandas as pd

from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import spacy
import textacy

from transformers import pipeline

import warnings
warnings.filterwarnings("ignore")


class LoadAndCleanData:

    def __init__(self, path):
        self.path = path            # Get the File Path from the directory


    def read_and_clean_data(self):
        review_df = pd.read_csv(self.path) 
        review_df = review_df.dropna().reset_index(drop=True)
        df = review_df['CleanedText']
        # df['CleanedText'].replace(regex=True, inplace=True, to_replace=r'[^0-9.\-]', value=r'')
        # review_df = review_df.drop(review_df['CleanedText'],axis = 0)
        return df


    # Remove Special Characters, Convert into the lower case and Stop Words & Apply Lemmatization

    def clean_text_and_lemmatize(self):

        stop_words = stopwords.words('english')    # list of Stop Words Excluding (not, no)
        stop_words.remove("not")
        stop_words.remove("no")

        lemmatizer = WordNetLemmatizer()        # Apply Lemmatization
        corpus_list = []

        df = self.read_and_clean_data()

        for i in tqdm(range(0, len(df))):
            review = re.sub('[^a-zA-Z]', ' ', str(df[i]))
            review = review.lower()
            review = review.split()
            
            review = [lemmatizer.lemmatize(word) for word in review if not word in stop_words]
            review = ' '.join(review)
            corpus_list.append(review)

        return corpus_list




class TopicClassifier:

    def __init__(self, data):
        self.data = data        # Get the clean data List

    # Remove Empty String from Corpus List

    def convert_empty_string_to_none(self):
        return [str(data or None) for data in self.data]
        

    # Get Keywords & Apply Part of Speech  

    def get_keywords_phreses(self):

        nlp = spacy.load("en_core_web_sm")
        key_phreses_list = []

        for data in tqdm(range(0, len(self.data))):
            doc = nlp(self.data[data])

            pattern = [[{"POS":"ADJ"}, {"POS":"NOUN"}] or [{"POS":"ADJ"}]]

            combined_pattern = textacy.extract.matches.token_matches(doc, patterns=pattern)

            combined_list = [str(word) for word in combined_pattern]

            str_list = ",".join(combined_list)
            key_phreses_list.append(str_list)
        # print(key_phreses_list)

        return key_phreses_list

# {"POS":"NOUN"}, {"POS":"PRON"}, {"POS":"VERB"}, {"POS":"ADV"}




class TypeClassifier:

    def __init__(self, data):
        self.data = data        # Get the clean Corpus List

    # Apply Zero Shot Classifier using hugging face

    def get_type_classifier_model(self, list):

        classifier = pipeline("zero-shot-classification", model='typeform/distilbert-base-uncased-mnli')
        result = classifier(sequences=self.data, candidate_labels=list, multi_class=True) 

        type_list = []
        for data in result:
            reviews = dict(zip(data['labels'], data['scores']))
            type_list.append(max(reviews, key = reviews.get))

        print(type_list)
        return type_list

    # candidate_labels = ['appreciation', 'information', 'complaint']




class SentimentClassifier:

    def __init__(self, data):
        self.data = data        # Get the list of Corpus

    # Apply Sentiment Classifier Model

    def get_sentiment_classifier_model(self):
        classifier = pipeline("sentiment-analysis")
        result = classifier(self.data)

        labels_list = []
        for data in result:
            labels_list.append(data['label'])
        
        print(labels_list)
        return labels_list