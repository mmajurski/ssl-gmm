import os

ifp = './models-cifar10'
# ifp = './models-cifar100'
# ifp = './models-ood'

fns = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
fns.sort()

idx = 0
while True:
    if len(fns) == 0:
        break

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
