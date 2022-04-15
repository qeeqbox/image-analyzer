#!/usr/bin/env python

'''
//  -------------------------------------------------------------
//  author        Giga
//  project       qeeqbox/image-analyzer
//  email         gigaqeeq@gmail.com
//  description   app.py (CLI)
//  -------------------------------------------------------------
//  contributors list qeeqbox/image-analyzer/graphs/contributors
//  -------------------------------------------------------------
'''
from warnings import filterwarnings
from os import environ, path

filterwarnings('ignore')
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
from json import dumps, JSONEncoder
from glob import glob
from PIL import Image
from time import sleep as ssleep
from requests import get, head
from bs4 import BeautifulSoup
from contextlib import suppress
from urllib.parse import urlparse, urljoin
from logging import getLogger, DEBUG, Formatter, StreamHandler
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from sys import stdout
from io import BytesIO
from asyncio import sleep, get_event_loop, Event, Queue
from aiohttp import web
from functools import wraps
from threading import Lock
from base64 import b64encode	
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from re import match as rematch
from operator import itemgetter
#from qbi import qbi

import tensorflow as tf
import numpy as np
import cv2

GLOBAL_SETTINGS = {'models_location':None, 'input_shape':[224,224], 'percentage':0.90, 'options':[], 'weights':{}, 'verbose':False}

def lock_function(f):
	lock = Lock()
	@wraps(f)
	def wrapper(self, *args, **kwargs):
		with lock:
			return f(self, *args, **kwargs)
	return wrapper

class ImageAnalyzer:
	def __init__(self, **kwargs):
		global GLOBAL_SETTINGS
		self.models = []
		self.models_location = kwargs.get('models_location', None) or GLOBAL_SETTINGS['models_location'] or path.join(path.dirname(__file__), '..', 'data')
		for filepath in glob(path.join(self.models_location,'*.h5')):
			matched = rematch(r"\[([\w \.]+)\]\[([\w \,\.]+)\]\[([\w ]+)\]\[([\w ]+)\]",path.basename(filepath))
			if matched:
				self.models.append({"name":matched[1], "info":matched[2],"categories":matched[3].split(" "), "number":matched[4], "model":load_model(filepath)})
		if len(self.models) > 0:
			self.input_shape = kwargs.get('input_shape', None) or GLOBAL_SETTINGS['input_shape']
			self.weights = kwargs.get('weights', None) or GLOBAL_SETTINGS['weights']
			self.percentage = kwargs.get('percentage', None) or GLOBAL_SETTINGS['percentage']
			self.options = kwargs.get('options', None) or GLOBAL_SETTINGS['options']
			self.verbose = kwargs.get('verbose', None) or GLOBAL_SETTINGS['verbose']
			self.headers={'User-Agent':'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:97.0) Gecko/20100101 Firefox/97.0'}
			self.models = sorted(self.models, key=itemgetter('number'))
			if self.verbose:
				self.log = getLogger('ImageAnalyzer')
				self.log.setLevel(DEBUG)
				sys_out_handler = StreamHandler(stdout)
				sys_out_handler.setLevel(DEBUG)
				sys_out_handler.setFormatter(Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
				self.log.addHandler(sys_out_handler)
				self.verbose and self.log.info('Models location {}'.format(self.models_location))
		else:
			exit()

	def check_url(self,url):
		with suppress():
			result = urlparse(url)
			if all([result.scheme, result.netloc]):
				self.verbose and self.log.info('{} is valid url'.format(url))
				return True
		return False

	def analyze_url(self,input_):
		type_ = None
		list_of_images = []
		temp_text_list = []
		list_of_temp_images = []
		self.verbose and self.log.info('Processing url {}'.format(input_))
		with suppress():
			res = head(input_)
			if res.headers.get('content-type',None) in ('image/jpeg', 'image/jpg', 'image/png'):
				type_ = 'link_to_image'
				img = Image.open(get(input_, stream = True,headers=self.headers).raw)
				img = img.convert('RGB')

				buffered = BytesIO()
				img.save(buffered, format='JPEG')
				base64_image = b64encode(buffered.getvalue()).decode('ASCII')

				np_image = np.array(img).astype('float32')/255
				np_image = tf.image.resize(np_image, self.input_shape)
				np_image = np.expand_dims(np_image, axis=0)
				image_s, ratio = self.process_image_feature(input_,img)
				list_of_images.append({'target':input_,'img':np_image, 'img_org':img,'img_s':image_s, 'ratio':ratio,'type':'image', 'base64':base64_image})
			else:
				type_ = 'website'
				content = ''
				text = ''
				if 'slow' not in self.options:
					self.verbose and self.log.info('Using fast option')
					content = get(input_,headers=self.headers).text
					soup = BeautifulSoup(content)
					[tag.extract() for tag in soup(['head','title','style', 'script', '[document]'])]
					temp_text_list = []
					for _ in soup.findAll(text=True):
						_ = _.strip()
						if _ not in temp_text_list and _ != '':
							temp_text_list.append(_)
				else:
					from galeodes import Galeodes
					if 'firefox' in self.options or 'chrome' not in self.options:
						self.verbose and self.log.info('Using slow option')
						g = Galeodes(browser='firefox', arguments=['--headless','--user-agent=\'{}\''.format(self.headers['User-Agent'])], options=None, implicit_wait=10, verbose=False)
						driver = g.setup_driver() 
						driver.get(input_)
						content = driver.page_source
						text = driver.find_element_by_tag_name('body').text
						soup = BeautifulSoup(content)
						[tag.extract() for tag in soup(['head','title','style', 'script', '[document]'])]
						temp_text_list = []
						for _ in soup.findAll(text=True):
							_ = _.strip()
							if _ not in temp_text_list and _ != '':
								temp_text_list.append(_)
						with suppress():
							x = driver.get_screenshot_as_png()
							img = Image.open(BytesIO(driver.get_screenshot_as_png()))
							img = img.convert('RGB')

							buffered = BytesIO()
							img.save(buffered, format='JPEG')
							base64_image = b64encode(buffered.getvalue()).decode('ASCII')

							#img.show()
							np_image = np.array(img).astype('float32')/255
							np_image = tf.image.resize(np_image, self.input_shape)
							np_image = np.expand_dims(np_image, axis=0)
							image_s, ratio = self.process_image_feature(input_,img)
							#list_of_images.append({'target':input_,'img':np_image,'img_org':img,'img_s':image_s, 'ratio':ratio, 'type':'screenshot','base64':base64_image})
						with suppress():
							driver.close()
						with suppress():
							driver.quit()
				for idx, image in enumerate(BeautifulSoup(content, 'html.parser').findAll('img')):
					if 'max_10' in self.options and idx > 10:
						break
					with suppress(Exception):
						if image['src'] not in list_of_temp_images:
							list_of_temp_images.append(image['src'])
							if image['src'].lower().startswith('http'):
								img = Image.open(get(image['src'], stream = True,headers=self.headers).raw)
								img = img.convert('RGB')

								buffered = BytesIO()
								img.save(buffered, format='JPEG')
								base64_image = b64encode(buffered.getvalue()).decode('ASCII')

								np_image = np.array(img).astype('float32')/255
								np_image = tf.image.resize(np_image, self.input_shape)
								np_image = np.expand_dims(np_image, axis=0)
								image_s, ratio = self.process_image_feature(image['src'],img)
								list_of_images.append({'target':image['src'],'img':np_image,'img_org':img,'img_s':image_s, 'ratio':ratio, 'type':'image', 'base64':base64_image})
		return type_,list_of_images, ' '.join(temp_text_list)

	def check_images(self,input_):
		list_of_images = []
		images = []
		with suppress():
			if path.isdir(input_):
				self.verbose and self.log.info('Processing dir {}'.format(input_))
				images = list(Path(input_).glob('*.jpg'))
				images.extend(list(Path(input_).glob('*.png')))
				images.extend(list(Path(input_).glob('*.gif')))
			elif path.isfile(input_):
				if input_.endswith(('.jpg','png','gif')):
					self.verbose and self.log.info('Processing file {}'.format(input_))
					images.append(input_)
			for image in images:
				with suppress():
					img = Image.open(image)
					img = img.convert('RGB')

					buffered = BytesIO()
					img.save(buffered, format='JPEG')
					base64_image = b64encode(buffered.getvalue()).decode('ASCII')

					np_image = np.array(img).astype('float32')/255
					np_image = tf.image.resize(np_image, self.input_shape)
					np_image = np.expand_dims(np_image, axis=0)
					image_s, ratio = self.process_image_feature(str(image),img)
					list_of_images.append({'target':str(image),'img':np_image, 'img_org':img,'img_s':image_s, 'ratio':ratio, 'type':'image','base64':base64_image})
		return list_of_images

	def process_image_feature(self,target, image):
		ratio = 0
		image_s = None
		self.verbose and self.log.info('Checking s_rate for {}'.format(target))
		with suppress():
			img = np.array(image)[:, :, ::-1]
			img_cvt = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
			s_rate_mask = cv2.inRange(img_cvt,np.array([0, 140, 85],np.uint8),np.array([255,180,140],np.uint8))
			s_rate = cv2.bitwise_and(img, img, mask = s_rate_mask)
			gray = cv2.cvtColor(s_rate, cv2.COLOR_BGR2GRAY)
			_,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			white = np.where(thresh==255)
			crop = s_rate[np.min(white[0]):np.max(white[0]), np.min(white[1]):np.max(white[1])]
			image_s = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_BGR2GRAY)
			black_pix = image_s.size - cv2.countNonZero(image_s)
			ratio = (100*(float(black_pix)/float(image_s.size)))
			if ratio != 100.0:
				ratio = 100-(100*(float(black_pix)/float(image_s.size)))
			image_s = Image.fromarray(image_s)
			image_s = image_s.convert('RGB')
			np_image = np.array(image_s).astype('float32')/255
			np_image = tf.image.resize(np_image, self.input_shape)
			np_image = np.expand_dims(np_image, axis=0)
		return np_image,ratio

	def find(self, list_, key, value):
		for i, dic in enumerate(list_):
			if dic[key] == value:
				return i
		return None

	def predict(self,type_, images):
		ret = []
		good = False
		temp_result = []
		predictions = []
		if len(images) > 0:
			with suppress():
				temp_images = [x['img'] for x in images]
				temp_images_full = [x for x in images if x['img'] is not None]
				for model in self.models:
					if model['name'] in self.options:
						good = True
						break
				for model in self.models:
					if model['name'] in self.options or good != True:
						self.verbose and self.log.info('Predicting [{}] for {} file[s]'.format(', '.join(model['categories']),len(temp_images)))
						predictions.append(model['model'].predict(np.vstack(temp_images)))
				for idx, val in enumerate(predictions[0]):
					overall = 0
					result = []
					categories_base = []
					with suppress():
						for idx_, prediction in enumerate(predictions):
							categories_base.append(self.models[idx_]['categories'][np.argmax(prediction[idx])])
						categories_base = Counter(categories_base)
						for temp_item in categories_base:
							result.append("[{}:{}]".format(temp_item, str(categories_base[temp_item])))
							for weight_item in self.weights:
								if temp_item == weight_item:
									overall += (categories_base[temp_item] * self.weights[temp_item])
						ret.append({'target':temp_images_full[idx]['target'],'category':', '.join(result),'weight':overall, 'base64': 'data:image/jpeg;base64,'+temp_images_full[idx]['base64']})

						print(ret)

		return ret

	#@lock_function
	def analyze(self,input_, options=None, remote=None):
		if remote:
			self.verbose and self.log.info('Processing request from {}'.format(remote))
		if options:
			self.options = options
		ret_ = {'urls':None,'images':None,'texts':None}
		if self.check_url(input_):
			type_, images, text = self.analyze_url(input_)
			ret = self.predict(type_, images)
			if len(ret) > 0:
				ret_['urls'] = ret
		else:
			with suppress():
				list_ = self.check_images(input_)
				ret = self.predict('image',list_)
				if len(ret) > 0:
					ret_['images'] = ret
		for item in ret_.copy():
			if ret_[item] == None:
				del ret_[item]

		with suppress():
			for cat in ['images','urls']:
				if cat in ret_:
					over_all = 0
					for item in ret_[cat]:
						over_all += item['weight']

					if over_all > 0:
						over_all = (over_all / (len(ret_[cat]) * 100)) * 100
						ret_['overall'] = round(over_all, 2), len(ret_[cat])
		return ret_

	def set_options(self, options):
		self.options = options

	#Private
	#def Call_QBI(self, obejct):
	#	pass

def process(json):
	ret = {}
	with suppress():
		imageanalyzer = ImageAnalyzer()
		imageanalyzer.set_options(json['options'])
		temp = imageanalyzer.analyze(json['target'], json['options'],json['remote'])
		if 'images' in temp:
			parsed_images = {'header':['id','category','weight','target','image'],'body':[],'info':'Images','name':'images-section'}
			for idx, item in enumerate(temp['images']):
				parsed_images['body'].append({'id':idx,'category':item['category'],'weight':item['weight'],'target':item['target'],'image':item['base64']})
			ret.update({'images':parsed_images})
		if 'urls' in temp:
			parsed_urls = {'header':['id','category','weight','target','image'],'body':[],'info':'Urls','name':'urls-section'}
			for idx, item in enumerate(temp['urls']):
				parsed_urls['body'].append({'id':idx,'category':item['category'],'weight':item['weight'],'target':item['target'],'image':item['base64']})
			ret.update({'urls':parsed_urls})
	return web.json_response(ret)

async def process_route(request):
	ret = {}
	with suppress():
		json = await request.json()
		json['remote'] = request.remote
		print(json)
		ret = await get_event_loop().run_in_executor(ProcessPoolExecutor(), process, json)
	return ret

async def home_route(request):
	index = ''
	models = []
	models_location = GLOBAL_SETTINGS['models_location'] or path.join(path.dirname(__file__), '..', 'data')
	for filepath in glob(path.join(models_location, '*.h5')):
		matched = rematch(r"\[([\w \.]+)\]\[([\w \,\.]+)\]\[([\w ]+)\]\[([\w ]+)\]",path.basename(filepath))
		if matched:
			models.append({"name":matched[1], "info":matched[2],"categories":matched[3].split(" "), "number":matched[4]})
	if len(models) > 0:
		with open(path.join(path.dirname(__file__), '..' ,'data', 'index.html'), 'r') as f:
			index = f.read()
		if '%< placeholder_1 >%' in index and '%< placeholder_2 >%' in index:
			render = []
			for item in models:
				render.append('{ group: \'Models\', value: \''+ item['name'] +'\', name: \''+item['info']+'\', disable: false}')
			index = index.replace('%< placeholder_1 >%', ', '.join(render) + ',')
			index = index.replace('%< placeholder_2 >%', models_location)
	if index:
		return web.Response(text=index,content_type='text/html')
	else:
		return web.Response(text="Something wrong..",content_type='text/html')

async def server(address, port):
	app1 = web.Application()
	app1.add_routes([web.post('/process', process_route, expect_handler = web.Request.json),web.get('/', home_route)])
	runner = web.AppRunner(app1)
	await runner.setup()
	await web.TCPSite(runner, address, port).start()
	await Event().wait()

def run_server(settings={}, address='0.0.0.0', port='8080'):
	GLOBAL_SETTINGS.update(**settings)
	get_event_loop().run_until_complete(server(address, port))

#run_server(settings={'input_shape':[224,224], 'percentage':0.90, 'options':[], 'weights': {'safe':50,'maybe':75,'unsafe':100,'safe_website':50,'unsafe_website':100}, 'verbose':False}, port='9090')