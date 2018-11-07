import pyopencl as cl
import numpy as np
from sys import platform

if platform == "darwin":
    VENDOR_NAME = "oclgrind"
elif platform == "win32":
    VENDOR_NAME = "nvidia"
else:
    VENDOR_NAME = "intel"
DEVICE_TYPE = cl.device_type.GPU if VENDOR_NAME == 'apple' or VENDOR_NAME == 'nvidia' or VENDOR_NAME == 'oclgrind' else cl.device_type.ACCELERATOR
print(DEVICE_TYPE, cl.device_type.GPU)
NN_T = np.float32
