import pyopencl as cl
import numpy as np
from sys import platform

if platform == "darwin":
    VENDOR_NAME = "apple"
else:
    VENDOR_NAME = "nvidia"
DEVICE_TYPE = cl.device_type.GPU if VENDOR_NAME == 'apple' or VENDOR_NAME == 'nvidia' else cl.device_type.ACCELERATOR

NN_T = np.float32
