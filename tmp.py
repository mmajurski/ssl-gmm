import os
import shutil
import json

import json

# ifp = '/home/mmajurski/github/ssl-gmm/models-cifar10'
ifp = '/home/mmajurski/github/ssl-gmm/models-ingest'
fns = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
fns.sort()

for fn in fns:
    config_fp = os.path.join(ifp, fn, 'config.json')
    # load config json file into dict
    with open(config_fp, 'r') as fh:
        config_dict = json.load(fh)

    if config_dict['last_layer'] == 'aa_gmm_d1':
        print(fn)
        config_dict['last_layer'] = 'kmeans'
        with open(config_fp, 'w') as fh:
            json.dump(config_dict, fh, ensure_ascii=True, indent=2)


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

