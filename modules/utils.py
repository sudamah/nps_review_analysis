import pandas as pd

from modules.services import LoadAndCleanData, TopicClassifier, SentimentClassifier, TypeClassifier


class LabelsToDataframe:

    def __init__(self, data):
        self.data = data

    def get_data_frame(self):

        path = '/home/heptagon/Desktop/nps_analysis/reviews_data.csv'
        load_data = LoadAndCleanData(path)

        load_data.clean_text_and_lemmatize()
