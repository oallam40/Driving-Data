# Libraries
import socketio
import eventlet
import dns
import cv2
import keras
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from flask import Flask
import base64
from io import BytesIO
from PIL import Image


# This script is set up to serve as a bridge between a driving simulator and a machine learning model that predicts steering angles based on camera images.
# It uses Flask for the web framework, Socket.IO for real-time communication, and TensorFlow (Keras) for running the predictive model.
# The script listens for telemetry data from the simulator, processes the received images, uses the model to predict the steering angle,
# and then sends steering and throttle commands back to the simulator.


sio = socketio.Server()
 
app = Flask(__name__) #'__main__'
speed_limit = 10

def img_preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img
    
# Define a handler for the "telemetry" event received from the client    
@sio.on('telemetry')
def telemetry(sid, data):
    # Extract the current speed of the car
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])                           
    steering_angle = float(model.predict(image))        # Predict the steering angle for the current frame
    throttle = 1.0 - speed/speed_limit                  # Calculate the throttle based on the current speed and the speed limit
    print('{} {} {}'.format(steering_angle, throttle, speed))       # Print the steering angle, throttle, and speed for debugging
    send_control(steering_angle, throttle)
 
 
 
# Define a handler for new connections
@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)          # On connection, send initial control commands to set steering and throttle to 0
 
# Function to emit steering and throttle commands to the client 
def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })
 
 
# Load the trained model
# Wrap the Flask app with Socket.IO communication
# Start the eventlet WSGI server and listen for connections
if __name__ == '__main__':
    model = load_model('model24.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)