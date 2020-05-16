# Simple Chatbot

The contents are as follow.

1. Simple Chatbot with Intent
    - [Simple Chatbot With RandomForest](https://medium.com/@akshay.mewada7/simple-chatbot-using-randoforest-3106ea2b523e)
    - Chatbot Webaap 1
2. Adding Entity to Simple chatbot
    - [Simple Named Entity Classification](https://medium.com/@akshay.mewada7/simple-named-entity-recognition-using-naive-bayes-classifier-2421f92c5b1e)
    - Chatbot Webapp 2
---
#### Simple Chatbot with Intent
The [article](https://medium.com/@akshay.mewada7/simple-chatbot-using-randoforest-3106ea2b523e) shows basic understaing of what is chatbot and how to create simple intent based chatbot. Also you can find full code in notebooks.  

Chatbot webapp 1 is the webapi example of Simple chatbot with random forest. The code found in chatbotwebpp_1.

---
#### Adding Entity to Simple Chatbot
Following the Intent based chatbot we need to add extra feature like information extraction form user text.

To achieve simple named entity classification the Simple Named Entity Recognition using Naive Bayes will help to understand. Also you can find full code in notebooks.   

Chatbot webapp 2 is the webapi example of Simple chatbot with intent and entity clssifier. The code found in chatbotwebpp_2.

---
### Installation Chatwebapps

#### Requiremnts packages
```
click==7.1.2
Flask==1.1.2
itsdangerous==1.1.0
Jinja2==2.11.2
joblib==0.14.1
MarkupSafe==1.1.1
nltk==3.5
numpy==1.18.4
PyYAML==5.3.1
regex==2020.5.14
scikit-learn==0.20.3
scipy==1.4.1
threadpoolctl==2.0.0
tqdm==4.46.0
Werkzeug==1.0.1
```
*Note: if youre using latest scikit learn change the **jobilib** import in code. To save and load model.*

#### Train Model

#### chatwebapp_1

Training Data format chatdata.json
```
{
    "greet": [
        "hi",
        "hello",
        "hey",
        "hola"
    ],
    "goodbye": [
        "bye",
        "goodbye",
        "good bye"
    ],
}
```

```
python train.py
```
*Note: make sure chatdata.json is in same dir*
*If want to change the training data change follow the above format*

#### chatwebapp_2

Trainig Data Format data.yaml
```
greet:
  - text: hi
  - text: hello
  - text: hey 

greetask:
  - text: hi i am john
    entity:
    - pos: 3
      name: john
  - text: hi i am ranvijay
    entity:
    - pos: 3
      name: ranvijay
  - text: hello i am steve
    entity:
    - pos: 3
      name: steve
```
*Niote: If want to change the training data change follow the above format*

Uncomment last three line in **modeltrainer.py**. Run below command to train.
```
python modeltrainer.py
```
*If want to change the training data change follow the above format*

#### Run Application
```
python app.py
```
Check url for chat response http://localhost:5000/chat?text="User text here"

***Thank You. Feel free to modify and upgrade the code.***
