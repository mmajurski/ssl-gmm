import os

ifp = './models-all'

fns = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
fns.sort()

idx = 0
while True:
    fn = 'id-{:08d}'.format(idx)
    dst = os.path.join(ifp, fn)
    if os.path.exists(dst):
        a = fns.index(fn)
        del fns[a]
        idx += 1
        continue

    src = os.path.join(ifp, fns[-1])
    os.rename(src, dst)
    del fns[-1]
    idx += 1
    if len(fns) == 0:
        break
