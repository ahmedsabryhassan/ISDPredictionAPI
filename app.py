from flask import Flask, render_template, request, json
from keras.models import load_model

# import the necessary packages image
from collections import deque
import numpy as np
import pickle
import cv2
import os

# text
import nltk
from nltk.corpus import stopwords
from nltk.stem import *
import re

app = Flask(__name__)

# load the trained model and label binarizer from disk
modelMedia = load_model('model/resnet50_Model.h5')
modelMedia.make_predict_function()
lb = pickle.loads(open('output/lb.pickle', "rb").read())

# initialize the image mean for mean subtraction along with the
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")


modelText = load_model('./model/textModel.h5')
modelText.make_predict_function()

def predict_Image(imgg):
	image = cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB)
	frame = cv2.resize(image, (224, 224)).astype("float32")
	# Frame classification inference and rolling prediction averaging come next:
	preds = modelMedia.predict(np.expand_dims(frame, axis=0))[0]
	i = np.argmax(preds)
	label = lb.classes_[i]
	return label

def predict_Video(vs):
	#initilizations
	Q = deque(maxlen=64) # this is utilizing our prediction averaging algorithm defualt is 128 changing it to 1 better
	
	# the start time is 0
	time = 0
	time_between_frames = 500 # millisecond

	results = None

	(W, H) = (None, None)
	# loop over frames from the video file stream
	while True:
		vs.set(cv2.CAP_PROP_POS_MSEC, time)
		# read the next frame from the file
		(grabbed, frame) = vs.read()
		# if the frame was not grabbed, then we have reached the end
		# of the stream
		if not grabbed:
			break
		# if the frame dimensions are empty, grab them
		if W is None or H is None:
			(H, W) = frame.shape[:2]
		####### --------------------- preprocess -------------
		# converting the from BGR to RGB
		# ordering, resize the frame to a fixed 224x224, and then
		# perform mean subtraction
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = cv2.resize(frame, (224, 224)).astype("float32")
		frame -= mean
		
		# Frame classification inference and rolling prediction averaging come next:
		preds = modelMedia.predict(np.expand_dims(frame, axis=0))[0]
		Q.append(preds)
		time += time_between_frames
	# perform prediction averaging over the current history of
	# previous predictions
	results = np.array(Q).mean(axis=0)
	i = np.argmax(results)
	label = lb.classes_[i]
	vs.release()
	return (label)

def predict_Text(data):
	if(data != ''):
		text=PorterStemmer()
		stop_words=stopwords.words("english")
		corpusdata = []
		for i in range (0, len(data)):
			text=re.sub('^\s+',"",str(data[i])) # remove space 
			text=re.sub('\s+$',"",str(data[i])) 
			text=re.sub('\s\d+\s',"",str(data[i]))
			text=re.sub('\s[a-zA-Z0-9]\s',"",str(data[i])) 
			text=re.sub('[^\w\s]',"",str(data[i]))  ## remove punc

			text = text.lower()
			text = text.split()
			#text = [lem.stem(word) for word in text if word not in stop_words]
			text = ' '.join(text)
			corpusdata.append(text)
		print(text)
		print(corpusdata)
		print(modelText.predict(corpusdata))
		return modelText.predict(corpusdata)

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/postPredicting", methods = ['GET', 'POST'])
def get_output():
	media_path = []
	mediaPrediction = []
	data = None
	if request.method == 'POST':
		for media in request.files.getlist('media'):
			media_path.append("static/" + media.filename)
			media.save(media_path[-1])
			if(media.content_type.split('/')[0] == 'image'):
				img = cv2.imread(media_path[-1])
				mediaPrediction.append({'Prediction':predict_Image(img), 'Type':'Image', 'Name': media.filename})
			elif(media.content_type.split('/')[0] == 'video'):
				vs = cv2.VideoCapture(media_path[-1])
				mediaPrediction.append({'Prediction':predict_Video(vs), 'Type':'Video', 'Name':media.filename})
				vs.release()
			os.remove(media_path[-1]) # deleting the image after predicting.
		data = {
			'Type':'post',
			'MediaPrediction': mediaPrediction if len(mediaPrediction) else 'NoMedia',
			'TextPrediction':'textPrediction'
		}
	else:
		data = {
			'working':'yes'
		}
	response = app.response_class(
		response=json.dumps(data),
		status=200,
		mimetype='application/json'
	)
	return response

if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True, host='0.0.0.0', port=8000)