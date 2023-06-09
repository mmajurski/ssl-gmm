import os
import shutil


ifp = '/home/mmajursk/github/ssl-gmm/models-20230604'
fns = [fn for fn in os.listdir(ifp) if fn.startswith('fix')]
fns.sort()

ofp = '/home/mmajursk/github/ssl-gmm/models-optuna2' 

image_id = 7
for fn in fns:
    src = os.path.join(ifp, fn)
    ofn = 'id-{:08d}'.format(image_id)
    shutil.move(src, os.path.join(ofp, ofn))
    image_id += 1

