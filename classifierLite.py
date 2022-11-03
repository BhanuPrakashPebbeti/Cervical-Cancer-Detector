import tensorflow as tf
import cv2
import numpy as np
from RoiCutter import create_model,cut_image

class classification_module:
  def __init__(self,path):
    self.path = path
    self.interpreter = tf.lite.Interpreter(model_path=self.path)
    self.interpreter.allocate_tensors()
    self.input_details = self.interpreter.get_input_details()
    self.output_details = self.interpreter.get_output_details()
    self.input_shape = (self.input_details[0]['shape'][1],self.input_details[0]['shape'][2])
    self.classes = {0:"negative",1:"positive"}
    self.obj_cut = create_model((self.input_shape[0],self.input_shape[1],3),c=[8,16,32,64])

  def preprocess(self,image):
    image = cv2.resize(image,(self.input_shape))/255
    image = np.expand_dims(np.array(image, dtype=np.float32), axis=0)
    return image

  def apply_roi_cutting(self,image):
    return cut_image(self.obj_cut,image,self.input_shape[0])

  def classify(self,image):
    self.interpreter.set_tensor(self.input_details[0]['index'], image)
    self.interpreter.invoke()
    output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
    return self.classes[output_data.argmax()],max(output_data)