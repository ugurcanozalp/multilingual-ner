from flask import Flask,render_template,url_for,request,jsonify
from multiner import MultiNerInferenceONNX

app = Flask(__name__)

ner = MultiNerInferenceONNX("ner_models/gold_model")

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
	text = request.form.get('text')
	result = ner(text)
	return jsonify(result)

if __name__=='__main__':
	app.run(debug=True, port='1080')