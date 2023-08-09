import os

ifp = './models-rng-seed'

fns = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
fns.sort()

idx = 0
while True:
    dst = os.path.join(ifp, 'id-{:08d}'.format(idx))
    if os.path.exists(dst):
        idx += 1
        continue

    src = os.path.join(ifp, fns[-1])
    os.rename(src, dst)
    del fns[-1]
    idx += 1
    if len(fns) == 0:
        break
