import tensorflow as tf
import tflite_runtime.interpreter as tflite
import cv2
import numpy as np
from RoiCutter import create_model, cut_image_lite

class classification_module:
  def __init__(self,path):
    self.path = path
    self.interpreter = tf.lite.Interpreter(model_path=self.path)
    self.interpreter.allocate_tensors()
    self.input_details = self.interpreter.get_input_details()
    self.output_details = self.interpreter.get_output_details()
    self.input_shape = (self.input_details[0]['shape'][1],self.input_details[0]['shape'][2])
    self.classes = {0:"negative",1:"positive"}
    self.interpreter_obj = tf.lite.Interpreter(model_path="/home/pi/app/models/obj_cut_lite.tflite")
    self.interpreter_obj.allocate_tensors()
    self.input_details_obj = self.interpreter_obj.get_input_details()
    self.output_details_obj = self.interpreter_obj.get_output_details()
    self.input_shape_obj = (self.input_details_obj[0]['shape'][1],self.input_details_obj[0]['shape'][2])

  def preprocess(self,image):
    image = cv2.resize(image,(self.input_shape))/255
    image = np.expand_dims(np.array(image, dtype=np.float32), axis=0)
    return image

  def apply_roi_cutting(self,image):
    image_roi = cv2.resize(image,(self.input_shape))/255
    image_roi = np.expand_dims(np.array(image_roi, dtype=np.float32), axis=0)
    self.interpreter_obj.set_tensor(self.input_details_obj[0]['index'], image_roi)
    self.interpreter_obj.invoke()
    output_data = self.interpreter_obj.get_tensor(self.output_details_obj[0]['index'])
    return cut_image_lite(output_data,image,self.input_shape[0])

  def classify(self,image):
    with tf.GradientTape() as tape:
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[1]['index'])
        last_conv_layer_output = self.interpreter.get_tensor(self.output_details[0]['index'])
        pred_index = output_data.argmax()
        class_channel = output_data[:, pred_index]
    print(class_channel,last_conv_layer_output)
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    print(heatmap.shape)
    return self.classes[output_data.argmax()],max(output_data)