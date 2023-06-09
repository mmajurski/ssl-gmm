import logging
logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                        handlers=[logging.StreamHandler()])

import os
os.environ['NUMEXPR_MAX_THREADS'] = "40"
import json
import optuna
from argparse import Namespace

import train


def objective(trial):

    i = 0
    ofp = os.path.join('models-optuna2/id-{:08d}'.format(i))
    while not os.path.exists(ofp):
        i += 1
        ofp = os.path.join('models-optuna2/id-{:08d}'.format(i))

    json_file_path = os.path.join(ofp, 'config.json')
    with open(json_file_path) as json_file:
        full_config_dict = json.load(json_file)

    # creating dictionary from stats.json file
    json_file_path = os.path.join(ofp, 'stats.json')
    with open(json_file_path) as json_file:
        result_dict = json.load(json_file)


    args_dict = dict()
    args_dict['arch'] = 'wide_resnet'
    args_dict['num_workers'] = 16
    args_dict['output_dirpath'] = ofp
    args_dict['batch_size'] = 64
    args_dict['learning_rate'] = trial.suggest_float("learning_rate", full_config_dict['learning_rate'], full_config_dict['learning_rate'])
    args_dict['tau'] = 1.0
    args_dict['tau_method'] = trial.suggest_categorical("tau_method", [full_config_dict['tau_method']])
    args_dict['mu'] = 7
    args_dict['loss_eps'] = 1e-4
    args_dict['patience'] = 50
    args_dict['strong_augmentation'] = False
    args_dict['debug'] = False
    args_dict['num_epochs'] = None
    args_dict['embedding_dim'] = trial.suggest_int("embedding_dim", full_config_dict['embedding_dim'], full_config_dict['embedding_dim'])
    args_dict['weight_decay'] = 5e-4
    args_dict['cycle_factor'] = 2.0
    args_dict['num_lr_reductions'] = 2
    args_dict['lr_reduction_factor'] = 0.2
    args_dict['nb_reps'] = 128
    args_dict['use_ema'] = trial.suggest_categorical("use_ema", [full_config_dict['use_ema']])
    args_dict['ema_decay'] = trial.suggest_float("ema_decay", full_config_dict['ema_decay'], full_config_dict['ema_decay'])
    args_dict['pseudo_label_threshold'] = trial.suggest_float("pseudo_label_threshold", full_config_dict['pseudo_label_threshold'], full_config_dict['pseudo_label_threshold'])
    args_dict['num_classes'] = 10
    args_dict['num_labeled_datapoints'] = 250
    args_dict['optimizer'] = 'adamw'
    args_dict['trainer'] = 'fixmatch-gmm'
    args_dict['last_layer'] = trial.suggest_categorical("last_layer", [full_config_dict['last_layer']])

    test_accuracy = result_dict['test_accuracy']
    
    shutil.move(ofp, ofp.replace('models-optuna2', 'models-optuna2-done'))

    return test_accuracy


def main():
    study_name = "study"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    if not os.path.exists('study-testing.db'):
        study = optuna.create_study(study_name=study_name, storage=storage_name, direction="maximize")
    else:
        study = optuna.load_study(study_name=study_name, storage=storage_name)

    study.optimize(objective, n_trials=1)

    # best_params = study.best_params
    print(study.best_trial)


if __name__ == "__main__":
    for i in range(100):
        if os.path.exists('study.db'):
            import shutil
            shutil.copy('study.db', 'study-bck.db')
        main()