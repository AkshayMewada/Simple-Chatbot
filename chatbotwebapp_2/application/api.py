from flask import request, jsonify
from application import webapp
from application.modelloader import IntentEntityModel
import random
import yaml

# Static Response
with open('application/data/response.yaml') as f:
    static_response = yaml.load(f)

# Operators
OPERATORS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/"
}

# IntentEntity Modell
model = IntentEntityModel(path='application/model')

class ChatGenerator:
    """
    ChatGenerator

    :Functions:

    get_response:
        :description:
        Gets user intention using 'model' i.e. IntentEntityModel.
        Generates the response on the basis of 'intent'.
        It uses the '_non_entity_response' and '_entity_response' 
        functions to generate response. 

        :param text: User text
        :return response: The chabot response to user text.

    _non_entity_response:
        :description:
        Generates the non entity based response.
        Where no information extraction is required.

        :param text: User text
        :return response: The chabot response to user text.

    _entity_response:
        :description:
        Generates the entity based response.
        Where information extraction is required.

        :param text: User text
        :return response: The chabot response to user text.
    """

    def get_response(self, text):
        """ Get Response """
        output = model.parse(text)
        if static_response.__contains__(output['intent']):
            if output['entity']:
                response = self._entity_response(intent=output['intent'],
                                                 entities=output['entity'])
            else:
                response = self._non_entity_response(intent=output['intent'])
        else:
            response = static_response['fallback']

        return response

    def _non_entity_response(self, intent):
        """ Non Entity Response """
        return random.choice(static_response[intent])

    def _entity_response(self, intent, entities):
        """ Entity Response """
        template = static_response['fallback']

        if intent == 'greetask':
            name = entities['name'][0]
            template = random.choice(static_response[intent])
            template = template.format(name=name.title())
        elif intent == 'domath':
            operator = entities['operator'][0]
            number1, number2 = tuple(entities['number'])
            template = random.choice(static_response[intent][operator])
            try:
                if operator == 'sub':
                    # sub num1 from num2
                    out = eval(number2+OPERATORS[operator]+number1)
                # add || mul || div  Number1 and Number2
                out = eval(number1+OPERATORS[operator]+number2)
                template = template.format(num1=number1,
                                           num2=number2,
                                           result=out)
            except:
                template = static_response['failed']
        return template


chat_gen_obj = ChatGenerator()


@webapp.route('/chat', methods=['GET'])
def chat():
    """ Get chatbot response View """

    try:
        try:
            text = request.args['text']
        except Exception as e:
            return jsonify({
                "status": False,
                "msg": "Bad Request"
            })

        # generate chat response
        response = chat_gen_obj.get_response(text)

        return jsonify({
            "status": True,
            "msg": response
        })

    except Exception as e:
        return jsonify({
            "status": True,
            "msg": "Something went wrong please try after sometime"
        })
