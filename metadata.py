import os
import json
import pandas as pd

import logging

logger = logging.getLogger()


class TrainingStats():
    def __init__(self):
        self.epoch_data = list()
        self.global_data = dict()
        self.metric_names = list()

    def add(self, epoch: int, metric_name: str, value):
        while len(self.epoch_data) <= epoch:
            self.epoch_data.append(dict())

        self.epoch_data[epoch]['epoch'] = epoch
        self.epoch_data[epoch][metric_name] = value

        logger.info('{}: {}'.format(metric_name, value))

    def update_global(self, epoch):
        if epoch > len(self.epoch_data):
            raise RuntimeError('Missing data at epoch {} in epoch stats'.format(epoch))

        epoch_data = self.epoch_data[epoch]
        for k, v in epoch_data.items():
            self.add_global(k, v)

    def add_global(self, metric_name: str, value):
        self.global_data[metric_name] = value

    def get(self, metric_name: str):
        data = list()
        for epoch_metrics in self.epoch_data:
            if metric_name not in epoch_metrics.keys():
                raise RuntimeError('Missing data for metric "{}" in epoch stats'.format(metric_name))
                # data.append(None)  # use this if you want to silently fail
            else:
                data.append(epoch_metrics[metric_name])
        return data

    def get_epoch(self, metric_name: str, epoch: int):
        if epoch > len(self.epoch_data):
            raise RuntimeError('Missing data for metric "{}" at epoch {} in epoch stats'.format(metric_name, epoch))
            # return None  # use this if you want to silently fail

        epoch_data = self.epoch_data[epoch]
        if metric_name not in epoch_data.keys():
            raise RuntimeError('Missing data for metric "{}" at epoch {} in epoch stats'.format(metric_name, epoch))
            # return None  # use this if you want to silently fail

        return epoch_data[metric_name]

    def get_global(self, metric_name: str):
        if metric_name not in self.global_data.keys():
            raise RuntimeError('Missing data for metric "{}" in global stats'.format(metric_name))
            # return None  # use this if you want to silently fail
        return self.global_data[metric_name]

    def export(self, output_folder: str):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # convert self.epoch_data into pandas dataframe
        df = pd.DataFrame(self.epoch_data)
        df.to_csv(os.path.join(output_folder, 'detailed_stats.csv'), index=False, encoding="ascii")

        # plot all metrics if its useful (its usually not)
        # fig = plt.figure(figsize=(8, 4), dpi=200)
        # for col in df.columns:
        #     if col == 'epoch':
        #         continue  # don't plot epochs against itself
        #     plt.clf()
        #     ax = plt.gca()
        #     x = df['epoch'].to_list()
        #     y = df[col].to_list()
        #     ax.plot(x, y, 'o-', markersize=5, linewidth=1)
        #     ax.set_xlabel('Epoch')
        #     ax.set_ylabel(col)
        #     plt.tight_layout()
        #     plt.savefig(os.path.join(output_folder, '{}.png'.format(col)))
        # plt.close(fig)

        with open(os.path.join(output_folder, 'stats.json'), 'w') as fh:
            json.dump(self.global_data, fh, ensure_ascii=True, indent=2)
