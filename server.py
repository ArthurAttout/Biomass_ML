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
from keras import backend as K
from tensorflow import Graph, Session
import mysql.connector

app = Flask(__name__)
app.debug = True
array_class = ["african_herbs","hay","marjorem","moss_green","moss_grey","rabbit_food","rosemary","sugar","tobacco"]
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
		
		graph1 = Graph()
		with graph1.as_default():
			session1 = Session()
			with session1.as_default():
				model_0 = load_model("/data/tera_1/partage/Biomass_ML/model.hdf5")
				for i in range (0,20):
					pr = model_0.predict(arrPred.reshape(1,img_width,img_height,3))[0]
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
	#classes_to_exclude = content['classes_to_exclude']
	id_model_target = content['model_target']
	
	print(url_host)
	print("Target model: {}".format(id_model_target))
	
	if id_model_target == '1':
		path_target = "/data/tera_1/partage/Biomass_ML/biomasse-1-2-3.hdf5"
	elif id_model_target == '2':
		path_target = "/data/tera_1/partage/Biomass_ML/biomasse-4-5-6.hdf5"
	elif id_model_target == '3':
		path_target = "/data/tera_1/partage/Biomass_ML/biomasse-7-8-9.hdf5"
	
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
		graph1 = Graph()
		with graph1.as_default():
			session1 = Session()
			with session1.as_default():
				model_target = load_model(path_target)
				pred = model_target.predict(arrPred.reshape(1,img_width,img_height,3))
				float_list = [float(i) for i in list(np.mean(pred, axis=0))]
				likely_class = int(np.argmax(np.mean(pred, axis=0)))
				
				response = {
					"predictions":float_list,
					"likely_class":likely_class,
					"certitude":max(float_list)
				}
				print("Identified with geoloc as {}".format(response))
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