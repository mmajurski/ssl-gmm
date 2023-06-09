import logging
logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                        handlers=[logging.StreamHandler()])

import os
os.environ['NUMEXPR_MAX_THREADS'] = "40"
import json
import shutil
import optuna
from optuna.distributions import FloatDistribution, CategoricalDistribution, IntDistribution

def get_params(ofp):

    json_file_path = os.path.join(ofp, 'config.json')
    with open(json_file_path) as json_file:
        full_config_dict = json.load(json_file)

    # creating dictionary from stats.json file
    json_file_path = os.path.join(ofp, 'stats.json')
    with open(json_file_path) as json_file:
        result_dict = json.load(json_file)

    args_dict = dict()
    # args_dict['arch'] = 'wide_resnet'
    # args_dict['num_workers'] = 16
    # args_dict['output_dirpath'] = ofp
    # args_dict['batch_size'] = 64
    args_dict['learning_rate'] = full_config_dict['learning_rate']
    # args_dict['tau'] = 1.0
    args_dict['tau_method'] = full_config_dict['tau_method']
    # args_dict['mu'] = 7
    # args_dict['loss_eps'] = 1e-4
    # args_dict['patience'] = 50
    # args_dict['strong_augmentation'] = False
    # args_dict['debug'] = False
    # args_dict['num_epochs'] = None
    args_dict['embedding_dim'] = full_config_dict['embedding_dim']
    # args_dict['weight_decay'] = 5e-4
    # args_dict['cycle_factor'] = 2.0
    # args_dict['num_lr_reductions'] = 2
    # args_dict['lr_reduction_factor'] = 0.2
    # args_dict['nb_reps'] = 128
    args_dict['use_ema'] = full_config_dict['use_ema']
    args_dict['ema_decay'] = full_config_dict['ema_decay']
    args_dict['pseudo_label_threshold'] = full_config_dict['pseudo_label_threshold']
    # args_dict['num_classes'] = 10
    # args_dict['num_labeled_datapoints'] = 250
    # args_dict['optimizer'] = 'adamw'
    # args_dict['trainer'] = 'fixmatch-gmm'
    args_dict['last_layer'] = full_config_dict['last_layer']

    if args_dict['last_layer'] not in ["aa_gmm", "aa_gmm_d1", "kmeans_layer"]:
        return None, None
    if args_dict['learning_rate'] < 1e-6 or args_dict['learning_rate'] > 1e-2:
        return None, None

    test_accuracy = result_dict['test_accuracy']

    return args_dict, test_accuracy


def get_trial_distributions():
    args_dict = dict()
    # args_dict['learning_rate'] = trial.suggest_float("learning_rate", 1e-6, 1e-2)
    # args_dict['tau_method'] = trial.suggest_categorical("tau_method", ['fixmatch', 'mixmatch'])
    # args_dict['embedding_dim'] = trial.suggest_int("embedding_dim", 4, 128)
    # args_dict['use_ema'] = trial.suggest_categorical("use_ema", [True, False])
    # args_dict['ema_decay'] = trial.suggest_float("ema_decay", 0.99, 0.9999)
    # args_dict['pseudo_label_threshold'] = trial.suggest_float("pseudo_label_threshold", 0.9, 0.9999)
    # args_dict['last_layer'] = trial.suggest_categorical("last_layer", ["aa_gmm", "aa_gmm_d1", "kmeans_layer"])

    args_dict['learning_rate'] = FloatDistribution(1e-6, 1e-2)
    args_dict['tau_method'] = CategoricalDistribution(['fixmatch', 'mixmatch'])
    args_dict['embedding_dim'] = IntDistribution(4, 128)
    args_dict['use_ema'] = CategoricalDistribution([True, False])
    args_dict['ema_decay'] = FloatDistribution(0.99, 0.9999)
    args_dict['pseudo_label_threshold'] = FloatDistribution(0.9, 0.9999)
    args_dict['last_layer'] = CategoricalDistribution(["aa_gmm", "aa_gmm_d1", "kmeans_layer"])
    return args_dict

# args_dict['arch'] = 'wide_resnet'
# args_dict['num_workers'] = 16
# args_dict['output_dirpath'] = ofp
# args_dict['batch_size'] = 64
# args_dict['learning_rate'] = trial.suggest_float("learning_rate", 1e-6, 1e-2)
# args_dict['tau'] = 1.0
# args_dict['tau_method'] = trial.suggest_categorical("tau_method", ['fixmatch', 'mixmatch'])
# args_dict['mu'] = 7
# args_dict['loss_eps'] = 1e-4
# args_dict['patience'] = 50
# args_dict['strong_augmentation'] = False
# args_dict['debug'] = False
# args_dict['num_epochs'] = None
# args_dict['embedding_dim'] = trial.suggest_int("embedding_dim", 4, 128)
# args_dict['weight_decay'] = 5e-4
# args_dict['cycle_factor'] = 2.0
# args_dict['num_lr_reductions'] = 2
# args_dict['lr_reduction_factor'] = 0.2
# args_dict['nb_reps'] = 128
# args_dict['use_ema'] = trial.suggest_categorical("use_ema", [True, False])
# args_dict['ema_decay'] = trial.suggest_float("ema_decay", 0.99, 0.9999)
# args_dict['pseudo_label_threshold'] = trial.suggest_float("pseudo_label_threshold", 0.9, 0.9999)
# args_dict['num_classes'] = 10
# args_dict['num_labeled_datapoints'] = 250
# args_dict['optimizer'] = 'adamw'
# args_dict['trainer'] = 'fixmatch-gmm'
# args_dict['last_layer'] = trial.suggest_categorical("last_layer", ["aa_gmm", "aa_gmm_d1", "kmeans_layer"])


def main():
    study_name = "study"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    if not os.path.exists('study.db'):
        study = optuna.create_study(study_name=study_name, storage=storage_name, direction="maximize")
    else:
        study = optuna.load_study(study_name=study_name, storage=storage_name)

    fns = [fn for fn in os.listdir('models-optuna2') if fn.startswith('id-')]
    fns.sort()

    t = 10
    for fn in fns:
        fp = os.path.join('models-optuna2', fn)
        args_dict, test_accuracy = get_params(fp)
        dist_dict = get_trial_distributions()
        if args_dict is not None:
            i = 0
            ofp = os.path.join('models-optuna/id-{:08d}'.format(i))
            while os.path.exists(ofp):
                i += 1
                ofp = os.path.join('models-optuna/id-{:08d}'.format(i))

            shutil.move(fp, ofp)
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
    study_name = "study"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    if not os.path.exists('study.db'):
        return

    study = optuna.load_study(study_name=study_name, storage=storage_name)
    print(study.best_trial)


if __name__ == "__main__":
    print_best_study()
    # main()