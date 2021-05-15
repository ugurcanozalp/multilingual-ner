from flask import Flask,render_template,url_for,request,jsonify
from multiner import MultiNerInferONNX, MultiNerInfer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_folder', '-m', default="ner_models/gold_model")
parser.add_argument('--onnx', '-o', default=False, action='store_true')
args = parser.parse_args()

app = Flask(__name__)

if args.onnx:
	ner = MultiNerInferONNX(args.model_folder)
else:
	ner = MultiNerInfer(args.model_folder)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
	text = request.form.get('text')
	result = ner(text)
	return jsonify(result)

if __name__=='__main__':
	app.run(debug=False, port='1080')
	#import requests
	#text = "Flask is a micro web framework written in Python."
	#url= "http://127.0.0.1:1080/predict"
	#response = requests.post(url, data={'text':text})
	#print(response.json())