from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import math
import numpy as np
import tensorflow as tf
import h5py


interpreter_quant = tf.lite.Interpreter(model_path="Model.tflite")# TFlite file



interpreter_quant.allocate_tensors()
input_index = interpreter_quant.get_input_details()[0]['index']
output_index = interpreter_quant.get_output_details()[0]['index']


predictions = interpreter_quant.get_tensor("Node index number as observed in hdf5 file previously generated") ##Add index as a integer 
print(predictions.shape)
tmp = predictions.flatten()
np.savetxt("All the weight values.txt",tmp)#File with trained weight values that can be included into numpy array for later analysis.
