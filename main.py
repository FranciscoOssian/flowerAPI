from markupsafe import escape
from flask import Flask
from flask import request
import tensorflow as tf
import numpy as np
import validators
import requests

app = Flask(__name__)

model = tf.keras.models.load_model('./model.h5')
img_height = 180
img_width = 180
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']


@app.route("/calculate", methods=['GET'])
def index():

  url = request.args.get('url')

  if( not validators.url(url) ):
    return {
      'error': 'this is not a valid url',
      'input': url
    }
  
  img_data = requests.get(url).content
  
  file = open(
    "./imageToTeste.png",
    "wb"
  )
  file.write(img_data)
  file.close()

  img = tf.keras.utils.load_img(
    './imageToTeste.png', target_size=(img_height, img_width)
  )

  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0)  # Create a batch
  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])
  
  resp = (
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
  )
  
  return {
    'response': resp
  }

app.run(host='192.168.0.51', port=8080)
