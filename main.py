# importing  all the

from modules.services import LoadAndCleanData, TopicClassifier, SentimentClassifier, TypeClassifier
# from modules.utils import LabelsToDataframe

path = '/home/heptagon/Desktop/nps_review_analysis/movie_reviews.csv'

candidate_labels = ['appreciation', 'information', 'complaint']

load_data = LoadAndCleanData(path)
corpus_list = TopicClassifier(load_data.clean_text_and_lemmatize())
# sentiments_list = SentimentClassifier(load_data.clean_text_and_lemmatize())
# type_list = TypeClassifier(load_data.convert_empty_string_to_none())

# data_frame = LabelsToDataframe(path)

if __name__ == '__main__':

    # calling functions

    # data_frame.clean_text_and_lemmatize()

    load_data.clean_text_and_lemmatize()
    corpus_list.get_keywords_phreses()
    # sentiments_list.get_sentiment_classifier_model()
    # type_list.get_type_classifier_model(candidate_labels)
