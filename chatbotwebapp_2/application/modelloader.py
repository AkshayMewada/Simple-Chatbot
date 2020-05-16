from application.modeltrainer import BaseProcessor
import pickle
import os
import numpy as np
from collections import defaultdict


class IntentEntityModel(BaseProcessor):
    """
    IntentEntityModel

    :param _dic_vec: Dictionary Vectorizer Model
           _tfidf_vec: Term Frequency Inverse Document Frequecny Model
           _intent_clf: Intent Classifier
           _entity_clf: Entity Classifier

    :Functions:
    parse:
        :description:
        The parse follows the below steps
        
        - Using parent class function 'processed_text' the function generates
          The intent features and entity features for predction. 
          For details goto BaseProcessor Class.
        - Intent Classification
        - Entity Classification
        - Creating data to return in given schema format. 
    
        :param text: text to classify
        :return data: Dictionary withc schema
            {
                "intent": Intent Name,
                "entity":{
                    "Entity Name": [List of Values]
                }
            }

    """
    def __init__(self, path):
        super().__init__()
        self.path = path
        self._dic_vec = pickle.load(open(os.path.join(
            self.path, self.dic_vec_name), 'rb+'))
        self._tfidf_vec = pickle.load(open(os.path.join(
            self.path, self.tfidf_name), 'rb+'))
        self._intent_clf = pickle.load(open(os.path.join(
            self.path, self.intent_name), 'rb+'))
        self._entity_clf = pickle.load(open(os.path.join(
            self.path, self.entity_name), 'rb+'))

    def parse(self, text):
        data = {
            "intent": None,
            "entity": defaultdict(list)
        }

        int_feature, ent_feature = self.processed_text(text)
        prob_ = self._intent_clf.predict_proba(int_feature)[0]
        max_ = np.argmax(prob_)
        entity_prob = self._entity_clf.predict(ent_feature)
        text_list = text.split(' ')

        if prob_[max_] > 0.6:
            data['intent'] = self._intent_clf.classes_[max_]

            for i in np.where(entity_prob != 'O')[0]:
                data['entity'][entity_prob[i]].append(text_list[i])

        data['entity'] = dict(data['entity'])

        return data