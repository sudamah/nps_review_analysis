# importing  all the

from modules.services import LoadAndCleanData, TopicClassifier, SentimentClassifier, TypeClassifier
from modules.utils import LabelsToDataframe

# path = '/home/heptagon/Desktop/nps_review_analysis/review.csv'
path = '/home/heptagon/Desktop/nps_review_analysis/dataset/BankPanacea_detail_review.csv'

# Get Most 20 Keywords
# load_data = LoadAndCleanData(path)
# corpus_list = TopicClassifier(load_data.get_lemmatize_data())

# Get Data Frame
data_frame = LabelsToDataframe(path)

if __name__ == '__main__':

    # calling functions

    data_frame.get_data_frame()

    # corpus_list.get_most_keywords_list()
