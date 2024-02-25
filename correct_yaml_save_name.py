import os
import yaml

ifp = '/home/mmajurski/github/ssl-gmm/usb/config/classic_cv/aagmm'
fns = [fn for fn in os.listdir(ifp) if fn.endswith('.yaml')]
fns.sort()

for fn in fns:
    print(fn)
    # load yaml file and modify a the save_name key to match fn
    with open(os.path.join(ifp, fn), 'r') as fh:
        data = yaml.load(fh, Loader=yaml.FullLoader)
    data['save_name'] = fn.replace('.yaml', '')
    # save the modified yaml file
    with open(os.path.join(ifp, fn), 'w') as fh:
        yaml.dump(data, fh)


