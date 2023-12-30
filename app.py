# Importing essential libraries
from flask import Flask, render_template, request
import pickle

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk

path = "./model/"
filename = path+ 'restaurant-sentiment-mnb-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open(path+'cv-transform.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	if request.method == 'POST':
			message = request.form['message']
			data = [message]
			vect = cv.transform(data).toarray() # covert text to vector
			my_prediction = classifier.predict(vect)
			return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	print('hello app is starting..')
	app.run()


