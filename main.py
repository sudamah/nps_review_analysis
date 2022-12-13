# importing  all the

from modules.services import LoadAndCleanData, TopicClassifier, SentimentClassifier, TypeClassifier
from modules.utils import LabelsToDataframe

path = '/home/heptagon/Desktop/nps_review_analysis/review.csv'
#
# candidate_labels = ['appreciation', 'information', 'complaint']

# load_data = LoadAndCleanData(path)
# corpus_list = TopicClassifier(load_data.get_lemmatize_data())
# sentiments_list = SentimentClassifier(load_data.get_lemmatize_data())
# type_list = TypeClassifier(load_data.convert_empty_string_to_none())

data_frame = LabelsToDataframe(path)

if __name__ == '__main__':

    # calling functions

    data_frame.get_data_frame()

    # load_data.get_lemmatize_data()
    # corpus_list.get_keywords_phreses()
    # sentiments_list.get_sentiment_classifier_model()
    # type_list.get_type_classifier_model(candidate_labels)
