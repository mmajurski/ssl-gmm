import psutil
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torchvision.models.resnet as resnet
import copy
import imageio
from matplotlib import pyplot as plt

import utils
import lcl_models
import metadata
import flavored_wideresnet
import embedding_constraints




class Net(nn.Module):
    def __init__(self, dim, num_classes=10):
        super(Net, self).__init__()


        self.dim = dim
        self.num_classes = num_classes
        # ----------
        # The architecture
        # ----------
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, dim)
        self.fc3 = nn.Linear(dim, num_classes)
        self.count = 0

    def forward(self, x):
        # ----------
        # The architecture
        # ----------
        # with torch.no_grad():
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        y = self.fc3(x)

        if not self.training and self.dim == 2:
            cluster_assignment = torch.argmax(y, dim=-1)

            fig = plt.figure(figsize=(4, 4), dpi=400)
            xcoord = x[:, 0].detach().cpu().numpy().squeeze()
            ycoord = x[:, 1].detach().cpu().numpy().squeeze()
            c_ids = cluster_assignment.detach().cpu().numpy().squeeze()
            cmap = plt.get_cmap('tab10')
            for c in range(self.num_classes):
                idx = c_ids == c
                cs = [cmap(c)]
                xs = xcoord[idx]
                ys = ycoord[idx]
                plt.scatter(xs, ys, c=cs, alpha=0.1, s=8)
            plt.title('Epoch {}'.format(self.count))
            plt.savefig('feature_space_{:04d}.png'.format(self.count))
            self.count += 1
            plt.close()

        return y


def train(args, model, device, train_loader, optimizer, epoch, train_stats, num_classes, embedding_dim, embedding_constraint=None):
    model.train()

    # Setup loss criteria
    criterion = torch.nn.CrossEntropyLoss()
    loss_list = list()
    loss_emb_list = list()
    accuracy_list = list()

    embedding_criterion = torch.nn.MSELoss()
    if embedding_constraint is None:
        emb_constraint = None
    elif embedding_constraint == 'mean_covar':
        emb_constraint = embedding_constraints.MeanCovar()
    elif embedding_constraint == 'gauss_moment':
        emb_constraint = embedding_constraints.GaussianMoments(embedding_dim=embedding_dim, num_classes=num_classes)
    elif embedding_constraint == 'l2':
        emb_constraint = embedding_constraints.L2ClusterCentroid()
    else:
        raise RuntimeError("Invalid embedding constraint type: {}".format(embedding_constraint))

    nan_count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)

        embedding, logits = model(data)

        batch_loss = criterion(logits, target)
        if emb_constraint is not None:
            emb_constraint_l = emb_constraint(embedding, model.last_layer.centers, logits)
            emb_constraint_loss = embedding_criterion(emb_constraint_l, torch.zeros_like(emb_constraint_l))
            loss_emb_list.append(emb_constraint_loss.item())
            batch_loss += emb_constraint_loss

        if torch.isnan(batch_loss):
            nan_count += 1
            if nan_count > 0.5 * len(train_loader):
                raise RuntimeError("more than 25% of the batches were nan, terminating")

        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        loss_list.append(batch_loss.item())
        acc = torch.argmax(logits, dim=-1) == target
        accuracy_list.append(torch.mean(acc, dtype=torch.float32).item())

        if batch_idx % args.log_interval == 0:
            # log loss and current GPU utilization
            cpu_mem_percent_used = psutil.virtual_memory().percent
            gpu_mem_percent_used, memory_total_info = utils.get_gpu_memory()
            gpu_mem_percent_used = [np.round(100 * x, 1) for x in gpu_mem_percent_used]
            print('  batch {}/{}  loss: {:8.8g}  lr: {:4.4g}  cpu_mem: {:2.1f}%  gpu_mem: {}% of {}MiB'.format(batch_idx, len(train_loader), batch_loss.item(), optimizer.param_groups[0]['lr'], cpu_mem_percent_used, gpu_mem_percent_used, memory_total_info))

    train_stats.add(epoch, 'train_loss', np.nanmean(loss_list))
    train_stats.add(epoch, 'train_emb_loss', np.nanmean(loss_emb_list))
    train_stats.add(epoch, 'train_accuracy', np.nanmean(accuracy_list))


def test(model, device, test_loader, epoch, train_stats):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    loss_list = list()
    accuracy_list = list()


    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            embedding, logits = model(data)

            loss = criterion(logits, target)

            loss_list.append(loss.item())

            acc = torch.argmax(logits, dim=-1) == target
            accuracy_list.append(torch.mean(acc, dtype=torch.float32).item())

    test_loss = np.nanmean(loss_list)
    test_acc = np.nanmean(accuracy_list)
    train_stats.add(epoch, 'test_loss', test_loss)
    train_stats.add(epoch, 'test_accuracy', test_acc)
    print('Test set (linear): Average loss: {:.4f}, Accuracy: {}'.format(test_loss, test_acc))

    return test_acc


def main(args):

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 10,
                       'pin_memory': True,
                       'shuffle': True}
        # check if IDE is in debug mode, and set the args debug flag and set num parallel worker to 0
        import utils
        if utils.is_ide_debug():
            print("setting num_workers to 0")
            cuda_kwargs['num_workers'] = 0

        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # model = Net(args.dim, num_classes=args.num_classes).to(device)
    embedding_dim = 2
    nprefc = 0
    output_folder = './models/softmax-{}'.format(args.emb_constraint)
    os.makedirs(output_folder, exist_ok=True)

    model = flavored_wideresnet.WideResNetMajurski(num_classes=10, last_layer=args.last_layer, depth=16, width=1, embedding_dim=embedding_dim, num_pre_fc=nprefc, use_tanh=False, output_folder=output_folder)
    model.to(device)

    import cifar_datasets
    train_dataset = cifar_datasets.Cifar10(transform=cifar_datasets.Cifar10.TRANSFORM_STRONG_AUGMIX_TRAIN, train=True)
    test_dataset = cifar_datasets.Cifar10(transform=cifar_datasets.Cifar10.TRANSFORM_TEST, train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_kwargs['batch_size'] = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)


    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=False)
    import lr_scheduler
    plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=4, threshold=1e-3, max_num_lr_reductions=1)


    train_stats = metadata.TrainingStats()
    epoch = -1
    MAX_EPOCHS = 2000
    best_model = copy.deepcopy(model)

    while not plateau_scheduler.is_done() and epoch < MAX_EPOCHS:
        epoch += 1
        train(args, model, device, train_loader, optimizer, epoch, train_stats, num_classes=10, embedding_dim=embedding_dim, embedding_constraint=args.emb_constraint)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        test_acc = test(model, device, test_loader, epoch, train_stats)
        plateau_scheduler.step(test_acc)

        if plateau_scheduler.is_equiv_to_best_epoch:
            print('Updating best model with epoch: {} accuracy: {}'.format(epoch, test_acc))
            # print('Updating best model with epoch: {} accuracy: {}'.format(epoch, avg_test_accuracy))
            best_model = copy.deepcopy(model)

            # update the global metrics with the best epoch
            train_stats.update_global(epoch)

        train_stats.export(output_folder)  # update metrics data on disk
        train_stats.plot_all_metrics(output_folder)

        # build gif, and remove tmp files
        fns = [fn for fn in os.listdir(output_folder) if fn.startswith('embedding_space_')]
        fns.sort()
        if len(fns) > 0:
            fps = 2
            if len(fns) > 50:
                fps = 4
            if len(fns) > 100:
                fps = 8
            duration = 1000 * (1.0 / fps)  # ms
            with imageio.get_writer(os.path.join(output_folder, 'EMB.gif'), mode='I', duration=duration, loop=0) as writer:
                for filename in fns:
                    image = imageio.imread(os.path.join(output_folder, filename))
                    writer.append_data(image)

    train_stats.export(output_folder)  # update metrics data on disk
    train_stats.plot_all_metrics(output_folder)

    torch.save(best_model, os.path.join(output_folder, "model.pt"))
    torch.save(best_model.state_dict(), os.path.join(output_folder, "model-state-dict.pt"))

    # build gif, and remove tmp files
    fns = [fn for fn in os.listdir(output_folder) if fn.startswith('embedding_space_')]
    fns.sort()
    for fn in fns:
        os.remove(os.path.join(output_folder, fn))



if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--last-layer', type=str, default='fc')
    parser.add_argument('--emb-constraint', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    main(args=args)



