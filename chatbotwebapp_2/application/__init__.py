from flask import Flask

# Initailise APP
webapp = Flask(__name__)

from application import api
