from flask import Flask,render_template,url_for,request,jsonify,Markup,flash
from templates.ner_to_html import render_ner_html
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model_folder', '-m', default=os.path.join("ner_models", "gold_model"))
parser.add_argument('--onnx', '-o', default=False, action='store_true')
args = parser.parse_args()

app = Flask(__name__, template_folder='templates')

if args.onnx:
	from multiner.infer_onnx import MultiNerInferONNX
	ner = MultiNerInferONNX(args.model_folder, model_name="optimized.onnx")
else:
	from multiner import MultiNerInfer
	ner = MultiNerInfer(args.model_folder)

#https://github.com/ankitshaw/entity-recognition-flask-app/blob/master/entity_recogizer.py
@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	text = request.form.get('text')
	result = ner(text)
	return jsonify(result)

@app.route('/', methods=['POST'])
def predict_html():
	text = request.form['text']
	print(text)
	print(type(text))
	results = ner(text)
	svg = render_ner_html(text, results)
	svg = Markup(svg)
	flash(svg)
	return render_template('index.html')

if __name__=='__main__':
	app.config['SECRET_KEY'] = os.urandom(24).hex()
	app.run(debug=False, port='5000')
	#import requests
	#text = "Flask is a micro web framework written in Python."
	#url= "http://127.0.0.1:5000/predict"
	#response = requests.post(url, data={'text':text})
	#print(response.json())