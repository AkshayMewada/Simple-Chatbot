import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

# Reading chat data
with open("chatdata.json", 'r') as f:
    chat_data = json.load(f)

# Training Data
training_dict = {}

# Appeding chat data in Training dicotionary
for intent, question_list in chat_data.items():
    for question in question_list:
        training_dict[question] = intent

# Separating Features i.e questions and Labels i.e intents
feature = np.array(list(training_dict.keys()))
labels = np.array(list(training_dict.values()))

# Converting text to WordVector
tf_vec = TfidfVectorizer().fit(feature)
X = tf_vec.transform(feature).toarray()

# Reshaping labels to fit data 
# Depends on vesions of scikit 
y = labels  

# Uncomment if above doesnt work
# y = labels.reshape(-1, 1)




# Fitting model
rnn = RandomForestClassifier(n_estimators=200)
rnn.fit(X, y)

# dumpmodels
joblib.dump(rnn, "chat_intent_model.pkl")
joblib.dump(tf_vec, "tf_vec.pkl")
