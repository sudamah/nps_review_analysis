import pandas as pd

from modules.services import LoadAndCleanData, TopicClassifier, SentimentClassifier, TypeClassifier


class LabelsToDataframe:

    def __init__(self, data):
        self.data = data

    def get_data_frame(self):
        pass
