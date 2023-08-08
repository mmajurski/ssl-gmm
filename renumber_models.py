import os

ifp = './models-rng-seed'

fns = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
fns.sort()

idx = 0
for fn in fns:
    src = os.path.join(ifp, fn)
    dst = os.path.join(ifp, 'id-{:08d}'.format(idx))
    while os.path.exists(dst):
        idx += 1
        dst = os.path.join(ifp, 'id-{:08d}'.format(idx))
    os.rename(src, dst)