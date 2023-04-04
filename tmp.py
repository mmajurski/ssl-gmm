import os
import shutil
import numpy as np
import torch

a = torch.from_numpy(np.asarray([0.01, 0.09, 0.2, 0.7]))
b = torch.log(a)
c = torch.nn.functional.softmax(b)
delta = torch.abs(a - c)
print(delta)




