import os
import shutil


# ifp = '/home/mmajursk/github/ssl-gmm/models-20230604'
# fns = [fn for fn in os.listdir(ifp) if fn.startswith('fix')]
# fns.sort()
#
# ofp = '/home/mmajursk/github/ssl-gmm/models-optuna2'
#
# image_id = 7
# for fn in fns:
#     src = os.path.join(ifp, fn)
#     ofn = 'id-{:08d}'.format(image_id)
#     shutil.move(src, os.path.join(ofp, ofn))
#     image_id += 1
#


import math
import torch


def get_cosine_schedule_with_warmup(num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return _lr_lambda


fh = get_cosine_schedule_with_warmup(0, 2**20)
x = list()
y = list()
for i in range(2**20):
    x.append(i)
    y.append(fh(i))


import matplotlib.pyplot as plt

# Plot the line
plt.plot(x, y)
# Show the plot
plt.show()
