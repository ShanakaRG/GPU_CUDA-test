# Step 1: Check Pytorch (optional)
import torch
print("Cuda available: ", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name())


# # Step 2: Check Tensorflow
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# Step 3: Check Keras (optional)
from keras import backend as K
print(K.tensorflow_backend._get_available_gpus())
