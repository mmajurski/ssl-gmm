import os

# ifp = '/mnt/isgnas/home/mmajursk/usb/saved_models/classic_cv'
# fns = [fn for fn in os.listdir(ifp) if os.path.isdir(os.path.join(ifp, fn))]
# fns.sort()
#
# to_del = list()
# for fn in fns:
#     if not os.path.exists(os.path.join(ifp, fn, 'success.txt')):
#         to_del.append(fn)
#
# import shutil
# for fn in to_del:
#     shutil.rmtree(os.path.join(ifp, fn))



ifp = '/home/mmajurski/github/ssl-gmm/usb/saved_models/classic_cv/stl-10'
fns = [fn for fn in os.listdir(ifp) if os.path.isdir(os.path.join(ifp, fn))]
fns.sort()

for fn in fns:
    if not os.path.exists(os.path.join(ifp, fn, 'success.txt')):
        print(fn)
