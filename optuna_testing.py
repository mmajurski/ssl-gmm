import logging
logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                        handlers=[logging.StreamHandler()])

import os
os.environ['NUMEXPR_MAX_THREADS'] = "40"
import optuna
from argparse import Namespace

import train


def objective(trial):

    i = 0
    if not os.path.exists('models-optuna'):
        os.makedirs('models-optuna')
    ofp = os.path.join('models-optuna/id-{:08d}'.format(i))
    while os.path.exists(ofp):
        i += 1
        ofp = os.path.join('models-optuna/id-{:08d}'.format(i))

    args_dict = dict()
    args_dict['arch'] = 'wide_resnet'
    args_dict['num_workers'] = 16
    args_dict['output_dirpath'] = ofp
    args_dict['batch_size'] = 64
    args_dict['learning_rate'] = trial.suggest_float("learning_rate", 1e-6, 1e-2)
    args_dict['tau'] = 1.0
    args_dict['tau_method'] = trial.suggest_categorical("tau_method", ['fixmatch', 'mixmatch'])
    args_dict['mu'] = 7
    args_dict['loss_eps'] = 1e-4
    args_dict['patience'] = 50
    args_dict['strong_augmentation'] = False
    args_dict['debug'] = False
    args_dict['num_epochs'] = None
    args_dict['embedding_dim'] = trial.suggest_int("embedding_dim", 4, 128)
    args_dict['weight_decay'] = 5e-4
    args_dict['cycle_factor'] = 2.0
    args_dict['num_lr_reductions'] = 2
    args_dict['lr_reduction_factor'] = 0.2
    args_dict['nb_reps'] = 128
    args_dict['use_ema'] = trial.suggest_categorical("use_ema", [True, False])
    args_dict['ema_decay'] = trial.suggest_float("ema_decay", 0.99, 0.9999)
    args_dict['pseudo_label_threshold'] = trial.suggest_float("pseudo_label_threshold", 0.9, 0.9999)
    args_dict['num_classes'] = 10
    args_dict['num_labeled_datapoints'] = 250
    args_dict['optimizer'] = 'adamw'
    args_dict['trainer'] = 'fixmatch-gmm'
    args_dict['last_layer'] = trial.suggest_categorical("last_layer", ["aa_gmm", "aa_gmm_d1", "kmeans_layer"])

    args = Namespace(**args_dict)
    train_stats = train.train(args)
    # remove the file handle logger
    logging.getLogger().removeHandler(logging.getLogger().handlers[1])
    test_accuracy = train_stats.get_global('test_accuracy')

    return test_accuracy


def main():
    study_name = "study"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    if not os.path.exists('study.db'):
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