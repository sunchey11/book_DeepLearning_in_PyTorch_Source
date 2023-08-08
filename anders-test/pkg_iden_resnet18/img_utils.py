from torch import Tensor
from PIL import Image
import numpy as np
import os
import torchvision.transforms as transforms
 
debug = False
def d_print(s):
    if(debug):
        print(s)