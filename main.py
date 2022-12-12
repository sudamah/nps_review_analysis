# importing  all the

from modules.services import LoadAndCleanData, TopicClassifier, SentimentClassifier, TypeClassifier


path = '/home/heptagon/Desktop/nps_analysis/reviews_data.csv'

candidate_labels = ['appreciation', 'information', 'complaint']

load_data = LoadAndCleanData(path)
corpus_list = TopicClassifier(load_data.clean_text_and_lemmatize())
sentiments_list = SentimentClassifier(load_data.clean_text_and_lemmatize())
type_list = TypeClassifier(load_data.convert_empty_string_to_none())


if __name__ == '__main__':

    # calling functions

    load_data.clean_text_and_lemmatize()
    corpus_list.get_keywords_phreses()
    sentiments_list.get_sentiment_classifier_model()
    type_list.get_type_classifier_model(candidate_labels)
