import re
import pandas as pd

from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import spacy
import textacy

from transformers import pipeline

import itertools
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")


class LoadAndCleanData:

    def __init__(self, path):
        self.path = path            # Get the File Path from the directory

    def read_dataset(self):
        review_df = pd.read_csv(self.path)
        review_df = review_df.dropna().reset_index(drop=True)
        df = review_df['comments']
        return df

    # Get the clean text list fron Dataset

    def get_clean_data(self):

        df = self.read_dataset()
        clean_text = []
        for i in range(0, len(df)):
            review = re.sub('[^a-zA-Z]', ' ', str(df[i]))
            # removing paragraph numbers
            review = re.sub('[0-9]+.\t','',str(review))
            # removing new line characters
            review = re.sub('\n ','',str(review))
            review = re.sub('\n',' ',str(review))
            # removing apostrophes
            review = re.sub("'s",'',str(review))
            # removing hyphens
            review = re.sub("-",' ',str(review))
            review = re.sub("— ",'',str(review))
            # removing quotation marks
            review = re.sub('\"','',str(review))
            # removing salutations
            review = re.sub("Mr\.",'Mr',str(review))
            review = re.sub("Mrs\.",'Mrs',str(review))
            # removing any reference to outside review
            review = re.sub("[\(\[].*?[\)\]]", "", str(review))
            review = " ".join(re.split("\s+", review, flags=re.UNICODE))

            if review != ' ':
                clean_text.append(review)

        # print(clean_text)
        return clean_text

    # Remove Special Characters, Convert into the lower case and Stop Words & Apply Lemmatization

    def get_lemmatize_data(self):

        # list of Stop Words Excluding (not, no)
        stop_words = stopwords.words('english')
        stop_words.remove("not")
        stop_words.remove("no")

        lemmatizer = WordNetLemmatizer()        # Apply Lemmatization
        corpus_list = []

        data_list = self.get_clean_data()

        for i in tqdm(range(0, len(data_list))):
            review = str(data_list[i]).lower()
            review = review.split()

            review = [lemmatizer.lemmatize(
                word) for word in review if not word in stop_words]
            review = ' '.join(review)
            corpus_list.append(review)

        # print(corpus_list)
        return corpus_list


class TopicClassifier:

    def __init__(self, data):
        self.data = data        # Get the clean data List

    # Get Keywords & Apply Part of Speech

    def get_keywords_phreses(self):

        nlp = spacy.load("en_core_web_sm")
        key_phreses_list = []

        for data in tqdm(range(0, len(self.data))):
            doc = nlp(self.data[data])

            pattern = [[{"POS": "ADJ"}, {"POS": "NOUN"}] or [{"POS": "ADJ"}]]

            combined_pattern = textacy.extract.matches.token_matches(
                doc, patterns=pattern)

            combined_list = [str(word) for word in combined_pattern]

            str_list = ",".join(combined_list[:20])
            key_phreses_list.append(str_list)

        # print(key_phreses_list)
        return key_phreses_list

    # Get None from Empty String from Key Phreses List

    def convert_empty_string_to_none(self):
        return [str(sent or None) for sent in self.get_keywords_phreses()]

    # Remove None from Key Phreses List

    def remove_none(self):
        return [i for i in self.get_keywords_phreses() if i]


# {"POS":"NOUN"}, {"POS":"PRON"}, {"POS":"VERB"}, {"POS":"ADV"}

    def get_most_keywords_list(self):
        res = [i for i in self.get_keywords_phreses() if i]
        res = ",".join([str(item) for item in res]).split(',')

        temp = defaultdict(int)    # Get Most frequent 20 Keywords

        for sub in res:
            temp[sub] += 1
        out = (dict(itertools.islice(dict(temp).items(), len(res))))
        out = {k: v for k, v in sorted(
            out.items(), key=lambda item: item[1], reverse=True)}
        out = out.keys()

        print(list(out)[:20])
        return list(out)[:20]


class TypeClassifier:

    def __init__(self, data):
        self.data = data        # Get the clean Corpus List

    # Apply Zero Shot Classifier using hugging face

    def get_type_classifier_model(self, list):

        classifier = pipeline("zero-shot-classification",
                              model='typeform/distilbert-base-uncased-mnli')

        result = classifier(sequences=self.data,
                            candidate_labels=list, multi_class=True)

        type_list = []
        for data in result:
            reviews = dict(zip(data['labels'], data['scores']))
            type_list.append(max(reviews, key=reviews.get))

        # print(type_list)
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

        # print(labels_list)
        return labels_list
