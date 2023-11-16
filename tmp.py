import os
import shutil
import json

import json

# # ifp = '/home/mmajurski/github/ssl-gmm/models-cifar10'
# ifp = '/home/mmajurski/github/ssl-gmm/models-ingest'
# fns = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
# fns.sort()
#
# for fn in fns:
#     config_fp = os.path.join(ifp, fn, 'config.json')
#     # load config json file into dict
#     with open(config_fp, 'r') as fh:
#         config_dict = json.load(fh)
#
#     if config_dict['last_layer'] == 'aa_gmm_d1':
#         print(fn)
#         config_dict['last_layer'] = 'kmeans'
#         with open(config_fp, 'w') as fh:
#             json.dump(config_dict, fh, ensure_ascii=True, indent=2)


# ifp = '/home/mmajurski/github/ssl-gmm/models-cifar10'
# fns = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
# fns.sort()
#
# for fn in fns:
#     config_fp = os.path.join(ifp, fn, 'config.json')
#     # load config json file into dict
#     with open(config_fp, 'r') as fh:
#         config_dict = json.load(fh)
#
#     if not config_dict['clip_grad'] and config_dict['last_layer'] == 'aa_gmm':
#         print(fn)


import numpy as np
import skimage.io


# fns = ["00000001", "00000013", "00000021", "00000054"]
# fns = ["00000020", "00000040", "00000087", "00000120"]

for fn in fns:
    ifp = '/home/mmajurski/github/ssl-gmm/paper/figures/id-{}-tsne.jpg'.format(fn)
    img = skimage.io.imread(ifp)

    d = 200
    img2 = img[d:-d, d:-d, :]
    img2 = img2[:, 0:-100, :]
    skimage.io.imsave(ifp, img2)
#
# mask = np.sum(img, axis=2)
# mask = mask != (3*255)
# mask = mask.astype(np.int32)
#
# mask_v = np.sum(mask, axis=0)
# mask_h = np.sum(mask, axis=1)
# idx = np.argwhere(mask_v)
# miny = idx[0][0]
# maxy = idx[-1][0]
#
# idx = np.argwhere(mask_h)
# minx = idx[0][0]
# maxx = idx[-1][0]
#
# img2 = img[miny:maxy, minx:maxx, :]
# skimage.io.imsave(ifp.replace('.jpg', '_cropped.jpg'), img2)
