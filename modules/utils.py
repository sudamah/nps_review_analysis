# import pandas as pd

# from modules.services import LoadAndCleanData, TopicClassifier, SentimentClassifier, TypeClassifier


# class LabelsToDataframe:

#     def __init__(self, path):
#         self.path = path            # Get the File Path from the directory

#     def get_data_frame(self):

#         candidate_labels = ['appreciation', 'information', 'complaint']

#         load_data = LoadAndCleanData()
#         corpus_list = TopicClassifier(load_data.clean_text_and_lemmatize())
#         sentiments_list = SentimentClassifier(
#             load_data.clean_text_and_lemmatize())
#         type_list = TypeClassifier(load_data.convert_empty_string_to_none())

#         load_data.clean_text_and_lemmatize()
#         corpus_list.get_keywords_phreses()
#         sentiments_list.get_sentiment_classifier_model()
#         type_list.get_type_classifier_model(candidate_labels)
