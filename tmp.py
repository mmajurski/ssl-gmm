import os
import shutil


ifp = '/mnt/isgnas/home/mmajursk/gmm/models-20230425'
fns = [fn for fn in os.listdir(ifp) if fn.startswith('fix')]
fns.sort()

for fn in fns:
    a = os.path.join(ifp, fn, 'model.pt')

