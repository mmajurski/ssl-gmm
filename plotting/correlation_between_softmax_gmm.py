import os
import numpy as np
import json

ifp = '/Users/mmajursk/Downloads/gmm-data/models'
fns = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
fns.sort()

softmax = list()
gmm = list()

for fn in fns:
    with open(os.path.join(ifp, fn, 'stats.json')) as json_file:
        stats = json.load(json_file)
    softmax.append(stats['test_softmax_accuracy'])
    gmm.append(stats['test_gmm_accuracy'])

cc = np.corrcoef(softmax, gmm)[0,1]

from matplotlib import pyplot as plt
fig = plt.figure(figsize=(7, 5), dpi=100)
plt.plot(softmax, gmm, '.')
ax = plt.gca()
ax.set_xlabel("softmax accuracy")
ax.set_ylabel("gmm accuracy")
ax.set_title('Model Accuracy: Softmax vs GMM (cc={0:.3f})'.format(cc))
plt.tight_layout()
plt.savefig("softmax-gmm-correlation.png")