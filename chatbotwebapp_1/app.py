# Import Library
import json
import numpy as np
from flask import Flask, request, jsonify
from sklearn.externals import joblib

# Initialize flask app
flask_app = Flask(__name__)

# Loading trained models
intent_clf_model = joblib.load('chat_intent_model.pkl')
tf_vec_model = joblib.load('tf_vec.pkl')

# Reading resposne
with open('response.json') as f:
    response_dict = json.load(f)


def get_respone(query):
    """ Predict Intent and Intent Respone """
    processed_text = tf_vec_model.transform([query])
    intent_prob = intent_clf_model.predict_proba(processed_text)[0]
    max_ = np.argmax(intent_prob)

    # If intent is has less probablity
    if intent_prob[max_] < 0.6:
        botout = "Sorry I am not getting you...!"
    else:
        # Appedning response
        botout = response_dict[intent_clf_model.classes_[max_]]

    return botout


@flask_app.route('/chat', methods=['GET'])
def chat():
    """ Get Chat Respone View """
    try:
        query = request.args['text']

        # Get response
        botresp = get_respone(query)
        response = {
            "status": True,
            "response": botresp
        }

    except:
        # If failed to generate respone
        response = {
            "status": False,
            "response": ""
        }

    return jsonify(response)


# Running Flask App
if __name__ == "__main__":
    flask_app.run()
