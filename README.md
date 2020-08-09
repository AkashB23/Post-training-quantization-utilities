# Post-training-quantization-utilities
Includes helping programmes  to analyse post training quantization in TensorFlow.

#### Requirements
1. Tensorflow 1.15.2
2. python >2.7
3. Hdf5 viewer
##### `ViewWeights.py`
Providing the TFlite file as input the programme outputs a `.hdf5` file that includes all the layers in the CNN/DNN model in the indexed fashion along with `name of the node (i.e. operation involved)` and `shape (input and output tensor shape)` and incase of Optimized (quantized) TFlite file as input, include the `scale` and `zero-point(offset)`  values.

##### `GetTensor.py`
After creating the `.hdf5` file note down the index number for which you want to get trained weight values. provide the path to same TFlite file and specify the index notified in the above programme to get a `Text file` with weight values as Output. `Text file` can be loaded as a numpy array to futher continue the analysis and comparision with quantized values.
