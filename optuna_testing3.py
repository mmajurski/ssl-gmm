import logging
logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                        handlers=[logging.StreamHandler()])

import os
os.environ['NUMEXPR_MAX_THREADS'] = "16"
import json
import shutil
import optuna
from optuna.distributions import FloatDistribution, CategoricalDistribution, IntDistribution

def get_params(ofp):

    json_file_path = os.path.join(ofp, 'config.json')
    with open(json_file_path) as json_file:
        full_config_dict = json.load(json_file)
        full_config_dict = full_config_dict['py/state']
    if full_config_dict['model_architecture']['value'] != 'resnet18':
        return None, None

    # creating dictionary from stats.json file
    json_file_path = os.path.join(ofp, 'stats.json')
    with open(json_file_path) as json_file:
        result_dict = json.load(json_file)

    args_dict = dict()
    args_dict['batch_size'] = full_config_dict['batch_size']['value']
    args_dict['learning_rate'] = full_config_dict['learning_rate']['value']
    args_dict['strong_augmentation'] = False

    if args_dict['learning_rate'] < 1e-6 or args_dict['learning_rate'] > 1e-2:
        return None, None
    if args_dict['batch_size'] < 16:
        return None, None

    test_accuracy = result_dict['test_clean_MulticlassAccuracy']

    return args_dict, test_accuracy


def get_trial_distributions():
    args_dict = dict()

    args_dict['batch_size'] = IntDistribution(16, 128)
    args_dict['strong_augmentation'] = CategoricalDistribution([True, False])
    args_dict['learning_rate'] = FloatDistribution(1e-6, 1e-2)
    return args_dict


def main():
    study_name = "study-supervised-cifar10"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    if not os.path.exists('{}.db'.format(study_name)):
        study = optuna.create_study(study_name=study_name, storage=storage_name, direction="maximize")
    else:
        study = optuna.load_study(study_name=study_name, storage=storage_name)

    fns = [fn for fn in os.listdir('/home/mmajurski/data/trojai/vision-trojans/models-new/') if fn.startswith('id-')]
    fns.sort()

    t = 0
    for fn in fns:
        fp = os.path.join('/home/mmajurski/data/trojai/vision-trojans/models-new/', fn)
        args_dict, test_accuracy = get_params(fp)
        dist_dict = get_trial_distributions()
        if args_dict is not None:
            trial = optuna.trial.create_trial(
                params=args_dict,
                distributions=dist_dict,
                value=t,
            )

            study.add_trial(trial)
            t += 1

    # best_params = study.best_params
    print(study.best_trial)

def print_best_study():
    study_name = "study-supervised-cifar10"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    if not os.path.exists('{}.db'.format(study_name)):
        return

    study = optuna.load_study(study_name=study_name, storage=storage_name)
    print(study.best_trial)
    print(study.best_params)


if __name__ == "__main__":
    print_best_study()
    # main()