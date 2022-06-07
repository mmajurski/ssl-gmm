import os
import shutil
from argparse import Namespace
import json
import train

learning_rate_levels = [1e-1, 7e-2, 5e-2, 3e-2, 1e-2, 7e-3, 5e-3, 3e-3, 1e-3, 7e-4, 5e-4, 3e-4, 1e-4, 7e-5, 5e-5, 3e-5, 1e-5]
batch_size_levels = [64]
num_epoch_levels = [None, 100, 200, 500]
number_reps = 5

root_ofp = './models/'

# for lr in learning_rate_levels:
#     for bs in batch_size_levels:
#         for num_epochs in num_epoch_levels:
#             for rep in range(number_reps):
#
#                 fn = "lr:{}-bs:{}-ne:{}".format(lr, bs, num_epochs)
#                 ofp = os.path.join(root_ofp, fn)
#                 if os.path.exists(ofp):
#                     if os.path.exists(os.path.join(ofp, 'model.pt')):
#                         continue
#                     else:
#                         shutil.rmtree(ofp)
#                 os.makedirs(ofp)
#
#                 arch = 'resnet18'
#                 #arch = 'resnet34'
#                 #arch = 'resnext50_32x4d'
#                 args = Namespace(arch=arch, num_workers=4, output_filepath=ofp, batch_size=bs, learning_rate=lr, loss_eps=1e-4, num_epochs=num_epochs, early_stopping_epoch_count=10)
#
#                 train.train(args)




best_accuracy = 0.0
best_config = None
for lr in learning_rate_levels:
    for bs in batch_size_levels:
        for num_epochs in num_epoch_levels:
            for rep in range(number_reps):

                fn = "lr:{}-bs:{}-ne:{}".format(lr, bs, num_epochs)
                ofp = os.path.join(root_ofp, fn)
                if not os.path.exists(ofp) or not os.path.exists(os.path.join(ofp, 'stats.json')):
                    continue
                with open(os.path.join(ofp, 'stats.json')) as json_file:
                    stats_dict = json.load(json_file)

                if 'test_accuracy' in stats_dict.keys():
                    if stats_dict['test_accuracy'] > best_accuracy:
                        best_accuracy = stats_dict['test_accuracy']

                        with open(os.path.join(ofp, 'config.json')) as json_file:
                            config = json.load(json_file)
                        best_config = config

print("Best Configuration (test accuracy = {})".format(best_accuracy))
print(best_config)





