from celery import Celery
import sys
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import tensorflow as tf
import urllib.request
from PIL import Image
import string
import random
from io import BytesIO
import mysql.connector
import time


app = Celery('tasks', broker='amqp://guest@localhost//')

mydb = mysql.connector.connect(
	host="192.168.1.67",
	user="python",
	passwd="python",
	database="biomass_database"
)

@app.task
def handle_images(content):
	subfolder_name = content['biomass_name'].lower().replace(' ','_')
	url_images = content['url_images']
	
	full_path = "/data/tera_1/partage/dataset/train/{}".format(subfolder_name)
	
	def randomString(stringLength=10):
		"""Generate a random string of fixed length """
		letters = string.ascii_uppercase
		return ''.join(random.choice(letters) for i in range(stringLength))
	
	print ("Check if {} exists".format(full_path))
	if os.path.exists(full_path):
		print("Path does exist")
		
	else:
		print("Path does not exist. Creating.")
		os.makedirs("/data/tera_1/partage/dataset/train/{}".format(subfolder_name))
		os.makedirs("/data/tera_1/partage/dataset/test/{}".format(subfolder_name))
		os.makedirs("/data/tera_1/partage/dataset/final/{}".format(subfolder_name))
		
		cursor = mydb.cursor()
		query_insert_class = '''
			INSERT INTO biomass (name,path_dataset) 
			VALUES ('{}','{}');
		'''
		cursor.execute(query_insert_class.format(subfolder_name, full_path))
		mydb.commit()
			
	train_path = "/data/tera_1/partage/dataset/train/{}".format(subfolder_name)
	test_path = "/data/tera_1/partage/dataset/test/{}".format(subfolder_name)
	final_path = "/data/tera_1/partage/dataset/final/{}".format(subfolder_name)

	file_count_train = 0
	for _, _, filenames in os.walk(train_path):
		file_count_train += len(filenames)
		
	file_count_test = 0
	for _, _, filenames in os.walk(test_path):
		file_count_test += len(filenames)
		
	file_count_final = 0
	for _, _, filenames in os.walk(final_path):
		file_count_final += len(filenames)
		
	file_total = file_count_final + file_count_test + file_count_train
	if file_total == 0:
		file_total = 1
	
	print("Current repartition is :")
	print("{} test".format(file_count_test/file_total))
	print("{} train".format(file_count_train/file_total))
	print("{} final".format(file_count_final/file_total))
	
	## Balance datasets
	
	# Get deltas
	delta_train = 0.75 - (file_count_train/file_total) 
	delta_test = 0.15 - (file_count_test/file_total) 
	delta_final = 0.10 - (file_count_final/file_total) 
	
	print("Delta test : {}".format(delta_test))
	print("Delta train : {}".format(delta_train))
	print("Delta final : {}".format(delta_final))
	
	destination_path = ""
	
	# Download each image + save in path
	for url in url_images:
	
		if delta_train > delta_test and delta_train > delta_final :
			print("Delta train is biggest. Add to train")
			destination_path = train_path
			file_count_train += 1
		elif delta_test > delta_train and delta_test > delta_final :
			print("Delta test is biggest. Add to test")
			destination_path = test_path
			file_count_test += 1
		else :
			print("Delta final is biggest. Add to final")
			destination_path = final_path
			file_count_final += 1
			
		print("Retrieving image at URL {}".format(url))
		with urllib.request.urlopen(url) as response:
			im = Image.open(BytesIO(response.read()))
			path_save = "{0}/{1}.png".format(destination_path,randomString())
			
			im.save(path_save,"PNG")
			print("Image saved at {}".format(path_save))
			file_total += 1
			
			#Recompute deltas
			delta_train = 0.75 - (file_count_train/file_total) 
			delta_test = 0.15 - (file_count_test/file_total) 
			delta_final = 0.10 - (file_count_final/file_total)
			print("Delta test : {}".format(delta_test))
			print("Delta train : {}".format(delta_train))
			print("Delta final : {}".format(delta_final))
			
	## End balance
	
	if(file_count_train >= 100):
		print("New folder exceeds threshold. Adding ML_class")
		
		cursor = mydb.cursor()
		query_update_class = '''
			SET SQL_SAFE_UPDATES=0;
			SET @new_class = (select MAX(class_ML)+1 from biomass);
			UPDATE biomass SET class_ML = @new_class WHERE biomass.name='{}';
			SET SQL_SAFE_UPDATES=1;
		'''
		results = cursor.execute(query_update_class.format(subfolder_name),multi=True)
		for cur in results:
			print('cursor:', cur)
			if cur.with_rows:
				print('result:', cur.fetchall())
			
		mydb.commit()

@app.task
def train_model():
	
	nb_classes = 0
	epochs = 35
	for _, dirnames, _ in os.walk(path):
		nb_classes += len(dirnames)
	
	print("Starting script with {} classes & {} epochs".format(nb_classes,epochs))
	img_width, img_height = 250, 250

	train_data_dir = "/data/tera_1/partage/dataset/train"
	validation_data_dir = "/data/tera_1/partage/dataset/test"
	nb_train_samples = 2000
	nb_validation_samples = 800
	batch_size = 16

	if K.image_data_format() == 'channels_first':
		input_shape = (3, img_width, img_height)
	else:
		input_shape = (img_width, img_height,3)

	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=input_shape))
	model.add(Activation('relu'))

	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))

	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))

	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())

	model.add(Dense(64))
	model.add(Activation('relu'))

	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])

	# this is the augmentation configuration we will use for training
	train_datagen = ImageDataGenerator(
			rotation_range=90,
			width_shift_range=0.3,
			height_shift_range=0.3,
			rescale=1./255,
			shear_range=0.2,
			zoom_range=0.3,
			horizontal_flip=True,
			cval=255,
			fill_mode='constant')

	# this is the augmentation configuration we will use for testing:
	# only rescaling
	test_datagen = ImageDataGenerator(rescale=1. / 255)

	train_generator = train_datagen.flow_from_directory(
		train_data_dir,
		color_mode='rgb',
		shuffle=True,
		target_size=(img_width, img_height),
		batch_size=batch_size,
		class_mode='categorical')

	test_generator = test_datagen.flow_from_directory(
		validation_data_dir,
		shuffle=True,
		color_mode='rgb',
		target_size=(img_width, img_height),
		batch_size=batch_size,
		class_mode='categorical')

	model.summary()
	filepath='/data/tera_1/partage/dataset/biomasse-checkpoint.hdf5'
	checkpoint = ModelCheckpoint(filepath, verbose=1)
	callbacks_list = [checkpoint]

	history = model.fit_generator(
		train_generator,
		validation_data = test_generator,
		validation_steps = 100,
		steps_per_epoch = (100 * nb_classes),
		epochs=epochs,callbacks=callbacks_list)

	model.save('/data/tera_1/partage/dataset/biomasse.hdf5')

	return "OK"