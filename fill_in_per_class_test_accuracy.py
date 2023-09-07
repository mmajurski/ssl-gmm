import os
import json
import argparse
import torch

import train
import trainer_fixmatch
import metadata
import embedding_constraints



ifp = './models-all'
fns = [fn for fn in os.listdir(ifp) if fn.startswith('id-')]
fns.sort()

for fn in fns:
    # load stats json file
    json_file_path = os.path.join(ifp, fn, 'stats.json')
    with open(json_file_path) as json_file:
        stats_dict = json.load(json_file)
    if 'test_accuracy_per_class' in stats_dict.keys():
        continue

    print(fn)

    # load the config json file
    json_file_path = os.path.join(ifp, fn, 'config.json')
    with open(json_file_path) as json_file:
        config_dict = json.load(json_file)
    if 'ood_p' not in config_dict.keys():
        config_dict['ood_p'] = 0.0

    parser = argparse.ArgumentParser()
    for k, v in config_dict.items():
        parser.add_argument('--' + k, default=v)
    args = parser.parse_args()

    # load the test dataset
    _, _, _, test_dataset = train.setup(args)
    model = torch.load(os.path.join(ifp, fn, 'model.pt'))
    model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    model_trainer = trainer_fixmatch.FixMatchTrainer(args)
    train_stats = metadata.TrainingStats()

    if args.embedding_constraint is None or args.embedding_constraint.lower() == 'none':
        emb_constraint = None
    elif args.embedding_constraint == 'mean_covar':
        emb_constraint = embedding_constraints.MeanCovar()
    elif args.embedding_constraint == 'gauss_moment':
        emb_constraint = embedding_constraints.GaussianMoments(embedding_dim=args.embedding_dim, num_classes=args.num_classes)
    elif args.embedding_constraint == 'l2':
        emb_constraint = embedding_constraints.L2ClusterCentroid()
    else:
        raise RuntimeError("Invalid embedding constraint type: {}".format(args.embedding_constraint))


    model_trainer.eval_model(model, test_dataset, criterion, train_stats, "test", emb_constraint, 0, args)
    per_class_test_accuracy = train_stats.epoch_data[0]['test_accuracy_per_class']
    print(per_class_test_accuracy)

    stats_dict['test_accuracy_per_class'] = per_class_test_accuracy
    json_file_path = os.path.join(ifp, fn, 'stats.json')
    with open(json_file_path, 'w') as fh:
        json.dump(stats_dict, fh, ensure_ascii=True, indent=2)



