import json
import time
import os

OUTPUT_DIR = 'data'


# TODO (majurski), merge with metadata
class GMMLogger:

    def __init__(self, desc=None):
        self.data = []
        self.epoch = None
        if desc is None:
            desc = "Data logging began at:" + time.strftime("%Y-%m-%d-%H-%M-%S") 
        self.data.append({"description": desc})

    def new_epoch(self, epoch=0):
        self.epoch = epoch
        self.data.append(dict())
        self.data[-1]['epoch'] = self.epoch

    def add_epoch_data(self, key, value):
        self.data[-1][key] = value

    def get(self, metric_name: str):
        data = list()
        for epoch_metrics in self.data:
            if metric_name not in epoch_metrics.keys():
                if "description" not in epoch_metrics.keys():  # if it's part of data and not description
                    raise RuntimeError('Missing data for metric "{}" in epoch stats'.format(metric_name))
                # data.append(None)  # use this if you want to silently fail
            else:
                data.append(epoch_metrics[metric_name])
        return data

    def export(self, filename=None):
        if filename is None:
            filename = "gmmdata_" + time.strftime("%Y-%m-%d-%H-%M-%S")+".log"  #add time string with suitable filename

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        with open(os.path.join(OUTPUT_DIR, filename), 'w') as log_file:
            log_file.write(json.dumps(self.data, indent=2))