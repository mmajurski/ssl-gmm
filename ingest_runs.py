import os
import shutil




# ifp = "/home/mmajurski/github/ssl-gmm/models"
# fns = [fn for fn in os.listdir(ifp) if fn.startswith('ssl-')]
# fns.sort()
#
# for fn in fns:
#     cur_fp = os.path.join(ifp, fn)
#
#     model_fns = [f for f in os.listdir(cur_fp) if f.startswith('id-')]
#     model_fns.sort()
#
#     for k in range(len(model_fns)):
#         model_fn = model_fns[k]
#         s = os.path.join(cur_fp, model_fn)
#         d = os.path.join(cur_fp, "tmp-id-{:04d}".format(k))
#         shutil.move(s,d)
#
#     model_fns = [f for f in os.listdir(cur_fp) if f.startswith('tmp-id-')]
#     model_fns.sort()
#
#     for model_fn in model_fns:
#         s = os.path.join(cur_fp, model_fn)
#         d = os.path.join(cur_fp, model_fn.replace('tmp-', ''))
#         shutil.move(s, d)



ifp = "/home/mmajurski/Downloads/models"
ofp = "/home/mmajurski/github/ssl-gmm/models"
fns = [fn for fn in os.listdir(ifp) if fn.startswith('ssl-')]
fns.sort()

for fn in fns:
    src_fp = os.path.join(ifp, fn)
    dest_fp = os.path.join(ofp, fn)
    if not os.path.exists(dest_fp):
        os.makedirs(dest_fp)

    model_fns = [f for f in os.listdir(src_fp) if f.startswith('id-')]
    model_fns.sort()

    k = 0
    for model_fn in model_fns:
        while os.path.exists(os.path.join(dest_fp, "id-{:04d}".format(k))):
            k += 1
        s = os.path.join(src_fp, model_fn)
        d = os.path.join(dest_fp, "id-{:04d}".format(k))
        shutil.copytree(s, d)

