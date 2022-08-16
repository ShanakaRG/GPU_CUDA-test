# Step 1: Check Pytorch (optional)
import torch
print("Cuda available: ", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name())

#full test for torch 
import torch
import sys
print('A', sys.version)
print('B', torch.__version__)
print('C', torch.cuda.is_available())
print('D', torch.backends.cudnn.enabled)
device = torch.device('cuda')
print('E', torch.cuda.get_device_properties(device))
print('F', torch.tensor([1.0, 2.0]).cuda())

# # Step 2: Check Tensorflow
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# Step 3: Check Keras (optional)
from keras import backend as K
print(K.tensorflow_backend._get_available_gpus())




