import numpy as np
import yaml
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
from tqdm import tqdm


FILE_FORMAT = ['yaml', 'yml']


class Error(Exception):
    """ Base Exception """
    pass


class FileFormatError(Error):
    """ File Format Exception """

    def __init__(self, message):
        self.message = message


class BaseProcessor:
    """
    BaseProcessor

    :param 
        # Overwritten by subaclasses
        _dic_vec: Dictionary Vectorizer Model
        _tfidf_vec: Term Frequency Inverse Document Frequecny Model
        _intent_clf: Intent Classifier
        _entity_clf: Entity Classifier
        # Common Params
        intent_name: File to save Intent Classifier
        dic_vec_name: File to save Dictonary Vectorizer Model
        entity_name: File to save Entity Classifier
        tfidf_name: File to save Term Frequency Inverse Document Frequecny Model

    :Function:

    processed_text:
        :description:
        The process text preprocess the text input for intent and entity classifier.
        The TF-IDF model is used for intent features.
        The DicVect model is used for entity features. The input to DicVect is from
        _get_features function.
    
        :param text: To to preprocess    
        :return int_feature, ent_feature:  The intent, entity features
    
    _get_features:
        :description:
        The get features uses the schema given below for better entity extraction.
        schema = {
            "word": Current Word,
            "postag": Part of Speech of Current Word,
            "nextword": Next Word,
            "nextwordtag": Nex Word POS tag,
            "previousword": Previous Word,
            "previoustag": Previous Word POS tag,
        }
        :param index: current word index
               word: Word
               tokens: Token list of given word
        
        :return dic: Return the feature dictionary.
    """

    def __init__(self):
        self._dic_vec = None
        self._tfidf_vec = None
        self._intent_clf = None
        self._entity_clf = None
        self.intent_name = 'intent.pkl'
        self.dic_vec_name = 'dicvec.pkl'
        self.entity_name = 'entity.pkl'
        self.tfidf_name = 'tfidf.pkl'

    def processed_text(self, text):
        ent_test_list = []
        token = text.split(' ')
        for i, word in enumerate(token):
            ent_test_list.append(self._get_features(i, word, token))
        ent_feature = self._dic_vec.transform(ent_test_list)
        int_feature = self._tfidf_vec.transform([text])
        return int_feature, ent_feature

    def _get_features(self, index, word, tokens):
        prev_word = 'BOS'
        next_word = 'EOS'

        if len(tokens) > index + 1:
            next_word = tokens[index + 1]
        if index - 1 > 0:
            prev_word = tokens[index - 1]

        val, tag = pos_tag([word])[0]
        prev_word, prev_tag = pos_tag([prev_word])[0]
        next_word, next_tag = pos_tag([next_word])[0]
        dic = {
            "word": val,
            "postag": tag,
            "nextword": next_word,
            "nextwordtag": next_tag,
            "previousword": prev_word,
            "previoustag": prev_tag,
        }
        return dic


class ModelTrainer(BaseProcessor):
    """
    ModelTrainer

    :param
        _dic_vec: DictVectorizer model for entity features
        _tfidf_vec: TfidfVectorizer model for intent features
        _intent_clf: RandomForestClassifier for intent classification
        _entity_clf: BernoulliNB for entity classification
        _ent_feature: Entity features
        _ent_label: Entity Labels
        _int_feature: Intent features
        _int_label: Intent Labels
        path:  Path to save models

    :Function:

    _check_file_format: 
        :description:
        Checks for YML/YAML file format
        
        :param file: Filename
        :return status: Is valid or not

    load:
        :description:
        The function loads the yaml training file and generates the training data.
        The load uses the function _intent_entity_extractor, _entity_label_extract and
        _get_features from  BaseProcessor. To create features and label.

        :param file: Training file

    _intent_entity_extractor:
        :description:
        Intent Entity Extractor uses the TF-IDF for intent features and labels genration.
        While DicVect and _get_features for entity features and labels genration. 
        The _entity_label_extract helps for entity label extraction

        :param data: training file data.

    _entity_label_extract:
        :description:
        Following training data schema function used to extract entity labels from that scehma.
        Default entity label is 'O'.
    
        :param question_dict: Traning data dictionary
               token_pos: token postion defined in trainig data. i.e Entity value postion

    _persist_helper:
        :description:
        Model object is saved as pickle file

        :param filename:File name to save
               object_: Model to save.

    _persist_models:
        :description:
        Saves all four models and classifers.

    train:
        :description:
        Trains the Intent and Entity Model
    """
    def __init__(self, path):
        super().__init__()
        self._dic_vec = DictVectorizer()
        self._tfidf_vec = TfidfVectorizer()
        self._intent_clf = RandomForestClassifier(n_estimators=200)
        self._entity_clf = BernoulliNB(alpha=0.1, binarize=0.1)
        self._ent_feature = []
        self._ent_label = []
        self._int_feature = []
        self._int_label = []
        self.path = path

    def _check_file_format(self, file):
        if file.split('.')[1] in FILE_FORMAT:
            return True
        return False

    def load(self, file):
        file_foarmt = self._check_file_format(file)
        if not file_foarmt:
            raise FileFormatError("Only YML/YAML file is allowed")

        with open(file, 'r') as f:
            data = yaml.load(f)

        ent_train_list, int_train_dict = self._intent_entity_extractor(data)

        int_feature_arr = np.array(list(int_train_dict.keys()))
        int_labels_arr = np.array(list(int_train_dict.values()))
        self._tfidf_vec.fit(int_feature_arr)
        self._int_feature = self._tfidf_vec.transform(
            int_feature_arr).toarray()
        self._int_label = int_labels_arr

        self._dic_vec.fit(ent_train_list)
        self._ent_feature = self._dic_vec.transform(ent_train_list).toarray()

    def _intent_entity_extractor(self, data):
        ent_train_list = []
        int_train_dict = {}

        for intent, question_list in tqdm(data.items()):
            for question_dict in question_list:

                token = question_dict['text'].split(' ')
                int_train_dict[question_dict['text']] = intent

                for i, word in enumerate(token):
                    self._entity_label_extract(question_dict, i)
                    ent_train_list.append(self._get_features(i, word, token))

        return ent_train_list, int_train_dict

    def _entity_label_extract(self, question_dict, token_pos):
        try:
            for ent in question_dict['entity']:
                k, v = list(ent.items())[1]
                if ent['pos'] == token_pos:
                    self._ent_label.append(k)
                    break
            else:
                self._ent_label.append('O')
        except:
            self._ent_label.append('O')

    def _persist_helper(self, filename, object_):
        with open(os.path.join(self.path, filename), 'wb+') as f:
            pickle.dump(object_, f)

    def _persist_models(self):
        self._persist_helper(self.dic_vec_name, self._dic_vec)
        self._persist_helper(self.tfidf_name, self._tfidf_vec)
        self._persist_helper(self.entity_name, self._entity_clf)
        self._persist_helper(self.intent_name, self._intent_clf)

    def train(self):
        self._entity_clf.fit(self._ent_feature, self._ent_label)
        self._intent_clf.fit(self._int_feature, self._int_label)
        self._persist_models()

"""
# Traing Model Ucomment the code and run current file.
# Must have model folder.
# must have training file in yaml format with given schema example.
model = ModelTrainer(path='model')
model.load('data/data.yaml')
model.train()
"""