from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import warnings
from wordcloud import WordCloud, STOPWORDS


class mostKeywords:

    tokenizer = AutoTokenizer.from_pretrained(
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    model = AutoModel.from_pretrained(
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    def mean_pooling(model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embeding_calc(sentences):

        # tokenizing
        encoded_input = tokenizer(
            sentences, padding=True, truncation=True, return_tensors='pt')

        # passing though model
        with torch.no_grad():
            model_output = model(**encoded_input)

        sentence_embeddings = mean_pooling(
            model_output, encoded_input['attention_mask'])

        return sentence_embeddings

    def similarity_calc(input1, input2):
        result = embeding_calc([input1, input2])
        score = cosine_similarity(result[0].reshape(
            1, -1), result[1].reshape(1, -1))
        return score[0][0]

    def read_data():
        df = pd.read_csv(
            '/home/heptagon/Desktop/nps_review_analysis/key_phrases_new.csv')
        return df

    def keywords():
        all_keyword = []
        for key_str in tqdm(df.topic):
            #     print(key_str)
            splited_str = key_str.split(',')
            if len(splited_str) > 0:
                all_keyword += splited_str
        #     break

        freq_count = {}
        for key in all_keyword:
            if key in freq_count:
                freq_count[key] += 1
            else:
                freq_count[key] = 1

        unique_keywords = freq_count.keys()

        clustures = []
        used_keywords = []

        for i in tqdm(unique_keywords):
            current_custer = [i]
            
            if (i in used_keywords):
                continue
            
            used_keywords.append(i)
            
            
            for j in unique_keywords:
                if  (j in used_keywords):
                    continue
            
                current_score = similarity_calc(i,j)
                if current_score>0.7:
                    current_custer.append(j)
                    used_keywords.append(j)
                    
            clustures.append(current_custer)

        final_result = []
        for cp in clustures:
            cluster_key_count = {}
            for key_c in cp:
                cluster_key_count[key_c] = freq_count[key_c]
            
            items = []
            for i in list(tuple(cluster_key_count.keys())):
                items.append(i.split()[0])
            dict1 = {}
            for item in items:
                if not item in dict1:
                    dict1[item] = items.count(item)
            dict1 = sorted(dict1.keys(), reverse=True)[0]

            curr_cul_keys = list(cluster_key_count.keys())
            curr_cul_values = list(cluster_key_count.values())
            sum_of_similar_words = np.sum(curr_cul_values)
            highest_count_key = curr_cul_keys[np.argmax(curr_cul_values)]
            highest_count_value = np.max(curr_cul_values)
            final_result.append([cluster_key_count,sum_of_similar_words,highest_count_key,len(cluster_key_count), dict1])
                
        df = pd.DataFrame(final_result,columns=["similar_words_and_count","count_of_similar_words","most_used_similar_word","total_similar_words", "key_expressions"])
        sorted_df = df.sort_values('total_similar_words',ascending=False).reset_index(drop=True)