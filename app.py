from flask import Flask, render_template, request,url_for
#from sklearn.linear_model import LogisticRegression
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

#from sklearn.pipeline import Pipeline
from gensim.summarization import summarize

import spacy



# Web Scraping Pkg
from bs4 import BeautifulSoup
from urllib.request import urlopen

import joblib
import pickle
import string
import nltk
#________________________________________________________________________________________________________________________
# summarize

# load the spacy english model

# model load sentiment analysis:
sen_model = joblib.load('models/sentiment.pkl')

#model_complain
C_model = joblib.load('models/complain.pkl')

# news classifier
#n_clf = joblib.load('models/news_classifier.pkl')

app = Flask(__name__)


# home page
@app.route('/')
def index():
	return render_template('home.html')


# sentiment analysis
@app.route('/nlpsentiment')
def sentiment_nlp():
	return render_template('sentiment.html')


@app.route('/sentiment',methods = ['POST','GET'])
def sentiment():
	if request.method == 'POST':
		message = request.form['message']
		# Machine learning analysiser
		pred = sen_model.predict([message])

	return render_template('sentiment.html', prediction=pred)

#------------------------------------------------------------------------------------------------------------------------------

# complain classifier
@app.route('/nlpcomplain')
def complain_nlp():
	return render_template('complain.html')


@app.route('/complain',methods = ['POST','GET'])
def complain():
	if request.method == 'POST':
		message = request.form['message']
		# Machine learning analysiser
		pred = C_model.predict([message])

	return render_template('complain.html', prediction=pred)


# spam classifier
@app.route('/nlpspam')
def spam_nlp():
	return render_template('spam.html')

# spam classification

@app.route('/spam',methods= ['POST','GET'])
def spam():
	df= pd.read_csv("spam.csv", encoding="latin-1")
	df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
	df.columns = ['label', 'message']
	# Features and Labels
	
	X = df['message']
	y = pd.get_dummies(df['label'])  #Ham = 1, spam = 0
	y = y.iloc[:,1].values


	# Extract Feature With CountVectorizer
	cv = CountVectorizer()
	X = cv.fit_transform(X) # Fit the Data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	#Naive Bayes Classifier
	from sklearn.naive_bayes import MultinomialNB

	clf = MultinomialNB()
	clf.fit(X_train,y_train)
	clf.score(X_test,y_test)
	

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
		
	return render_template('spam.html',prediction=my_prediction)


#----------------------------------------------------------------------------------------------------------------------------------


#summerize


# Fetch Text From Url
def get_text(url):
	page = urlopen(url)
	soup = BeautifulSoup(page)
	fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
	return fetched_text

@app.route('/nlpsummarize')
def summarize_nlp():
	return render_template('summarize.html')

# summarize 
@app.route('/summarize',methods= ['POST','GET'])
def sum_route():
	if request.method == 'POST':
		message = request.form['message']
		sum_message = summarize(message)
		
	return render_template('summarize.html',original = message, prediction=sum_message)

#analyse url
@app.route('/analyze_url',methods=['GET','POST'])
def analyze_url():
	if request.method == 'POST':
		raw_url = request.form['raw_url']
		rawtext = get_text(raw_url)
		final_summary = summarize(rawtext)
		
	return render_template('summarize.html',ctext=rawtext,final_summary=final_summary)

#-----------------------------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
	app.run(debug=True)
