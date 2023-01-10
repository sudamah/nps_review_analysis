import pandas as pd
import numpy as np

from modules.services import LoadAndCleanData, TopicClassifier, SentimentClassifier, TypeClassifier


class LabelsToDataframe:

    def __init__(self, data):
        self.data = data            # Get the File Path from the directory

    def get_data_frame(self):

        candidate_labels = ['appreciation', 'information', 'complaint']

        load_data = LoadAndCleanData(self.data)
        corpus_list = TopicClassifier(load_data.get_lemmatize_data())
        sentiments_list = SentimentClassifier(
            corpus_list.remove_none())
        type_list = TypeClassifier(corpus_list.remove_none())

        df_final = pd.DataFrame(
            load_data.get_clean_data(), columns=['comments'])
        df_final['topic'] = corpus_list.convert_empty_string_to_none()

        df_final = df_final.replace(to_replace='None', value=np.nan).dropna()

        df_final['sentiment'] = sentiments_list.get_sentiment_classifier_model()
        df_final['type'] = type_list.get_type_classifier_model(
            candidate_labels)

        print(df_final)
        print('-'*110)

        print(df_final.to_csv(
            "/home/heptagon/Desktop/nps_review_analysis/dataset/BankPanacea_detail_review_topic.csv", index=False))

        return df_final
