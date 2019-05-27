#!/usr/bin/env python

#ML COMPONENT

from flask import Flask
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
from io import BytesIO
import json
import base64
import os, sys
import random
from keras.models import load_model
from flask import request
from tasks import handle_images
import urllib.request
import string
from werkzeug.serving import run_simple
import mysql.connector

app = Flask(__name__)
app.debug = True
array_class = ["african_herbs","hay","marjorem","moss_green","moss_grey","rabbit_food","rosemary","sugar","tobacco"]
model = load_model("/data/tera_1/partage/Biomass_ML/model.hdf5")
model._make_predict_function()
img_width, img_height = 250, 250

mydb = mysql.connector.connect(
	host="192.168.1.67",
	user="python",
	passwd="python",
	database="biomass_database"
)


@app.route('/identify', methods = ['POST'])
def identifyHandler():
	content = request.get_json()
	url_host = content['url']
	
	print(url_host)
	
	with urllib.request.urlopen(url_host) as response:

		# Resize original image
		im = Image.open(BytesIO(response.read()))
		width,height = im.size
		
		if(content['crop']):
			overlay_size = 150 * 8

			x1 = (width / 2) - (overlay_size / 2)
			y1 = (height / 2) - (overlay_size / 2)
			x2 = (width / 2) + (overlay_size / 2)
			y2 = (height / 2) + (overlay_size / 2)
			im = im.crop((x1,y1,x2,y2))
			
		im = im.resize((img_width, img_height))

		arr = np.array(im)
		arrPred = np.asarray([i / 255 for i in arr.reshape(187500)])
		pred = []
		
		for i in range (0,20):
			pr = model.predict(arrPred.reshape(1,img_width,img_height,3))[0]
			pred.append(pr)
		
		float_list = [float(i) for i in list(np.mean(pred, axis=0))]
		likely_class = int(np.argmax(np.mean(pred, axis=0)))
		certitude = float(1 - np.std(pred,axis=0)[np.argmax(np.mean(pred, axis=0))])
		print("{0}, certitude {1}".format(likely_class, certitude))
		
		response = {
			"predictions":float_list,
			"likely_class":likely_class,
			"certitude":certitude
		}
		return json.dumps(response)
		
@app.route('/identifyWithMask', methods = ['POST'])
def identifyMaskHandler():
	content = request.get_json()
	url_host = content['url']
	classes_to_exclude = content['classes_to_exclude']
	
	print(url_host)
	print("Class excluded : {}".format(classes_to_exclude))
	
	with urllib.request.urlopen(url_host) as response:

		# Resize original image
		im = Image.open(BytesIO(response.read()))
		width,height = im.size
		if(content['crop']):
			overlay_size = 150 * 8

			x1 = (width / 2) - (overlay_size / 2)
			y1 = (height / 2) - (overlay_size / 2)
			x2 = (width / 2) + (overlay_size / 2)
			y2 = (height / 2) + (overlay_size / 2)
			im = im.crop((x1,y1,x2,y2))
			
		im = im.resize((img_width, img_height))
		arr = np.array(im)
		arrPred = np.asarray([i / 255 for i in arr.reshape(187500)])
		pred = model.predict(arrPred.reshape(1,img_width,img_height,3))
		float_list = [float(i) for i in list(pred[0])]
		
		print("Base list is {}".format(float_list))
		
		for excluded in classes_to_exclude:
			float_list[excluded] = 0
		
		print("Elagued list is {}".format(float_list))
		
		renormalized = []
		for prediction in float_list:
			renormalized.append(prediction / np.sum(float_list))
		
		likely = renormalized.index(max(renormalized))
		print("Renormalized list is {}".format(renormalized))
		response = {
			"predictions":renormalized,
			"likely_class":likely,
			"certitude":confusion_matrix[likely]
		}
		return json.dumps(response)
		
		
#Expected body : 
#{
#	"biomass_name":"African herbs",
#	"url_images":[
#		'https://....',
#		'https://....',
#		'https://....',
#	]
#}

@app.route('/add_images', methods = ['POST'])
def add_images():
		
	content = request.get_json()
	print("Starting task")
	
	handle_images.delay(content)
		
	return "OK"
		
if __name__ == '__main__':
	app.run(host = '0.0.0.0',port=5001)