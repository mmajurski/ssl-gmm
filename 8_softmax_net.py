import psutil
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

import utils
import lcl_models
import metadata



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        dim = 10

        # ----------
        # The architecture
        # ----------
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, dim)


    def forward(self, x):
        # ----------
        # The architecture
        # ----------
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)

        return x


def train(args, model, device, train_loader, optimizer, epoch, train_stats):
    model.train()

    # Setup loss criteria
    criterion = torch.nn.CrossEntropyLoss()
    loss_list = list()
    accuracy_list = list()


    # grad_scale_dict = dict()

    # tmp_stats = metadata.TrainingStats()
    #
    # try:
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)

        output = model(data)
        batch_loss = criterion(output, target)
        if torch.isnan(batch_loss):
            print("nan loss")

        batch_loss.backward()
        optimizer.step()
        loss_list.append(batch_loss.item())
        acc = torch.argmax(output, dim=-1) == target
        accuracy_list.append(torch.mean(acc, dtype=torch.float32).item())

        if batch_idx % args.log_interval == 0:
            # log loss and current GPU utilization
            cpu_mem_percent_used = psutil.virtual_memory().percent
            gpu_mem_percent_used, memory_total_info = utils.get_gpu_memory()
            gpu_mem_percent_used = [np.round(100 * x, 1) for x in gpu_mem_percent_used]
            print('  batch {}/{}  loss: {:8.8g}  lr: {:4.4g}  cpu_mem: {:2.1f}%  gpu_mem: {}% of {}MiB'.format(batch_idx, len(train_loader), batch_loss.item(), optimizer.param_groups[0]['lr'], cpu_mem_percent_used, gpu_mem_percent_used, memory_total_info))

    train_stats.add(epoch, 'train_loss', np.nanmean(loss_list))
    train_stats.add(epoch, 'train_accuracy', np.nanmean(accuracy_list))


def test(model, device, test_loader, epoch, train_stats):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    loss_list = list()
    accuracy_list = list()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            target_size = target.size()

            output = model(data)
            loss = criterion(output, target)
            loss_list.append(loss.item())
            acc = torch.argmax(output, dim=-1) == target
            accuracy_list.append(torch.mean(acc, dtype=torch.float32).item())

    test_loss = np.nanmean(loss_list)
    test_acc = np.nanmean(accuracy_list)
    train_stats.add(epoch, 'test_loss', test_loss)
    train_stats.add(epoch, 'test_accuracy', test_acc)

    print('Test set: Average loss: {:.4f}, Accuracy: {}'.format(test_loss, test_acc))
    return test_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 2,
                       'pin_memory': True,
                       'shuffle': True}
        # check if IDE is in debug mode, and set the args debug flag and set num parallel worker to 0
        import utils
        if utils.is_ide_debug():
            print("setting num_workers to 0")
            cuda_kwargs['num_workers'] = 0

        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)


    model = Net().to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    import lr_scheduler
    plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=20, threshold=1e-4, max_num_lr_reductions=2)


    output_folder = './models-20230312/adam-softmax'
    train_stats = metadata.TrainingStats()
    epoch = -1
    MAX_EPOCHS = 2000
    best_model = copy.deepcopy(model)
    while not plateau_scheduler.is_done() and epoch < MAX_EPOCHS:
        epoch += 1
        train(args, model, device, train_loader, optimizer, epoch, train_stats)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        test_loss = test(model, device, test_loader, epoch, train_stats)
        plateau_scheduler.step(test_loss)

        if plateau_scheduler.is_equiv_to_best_epoch:
            print('Updating best model with epoch: {} loss: {}'.format(epoch, test_loss))
            best_model = copy.deepcopy(model)

            # update the global metrics with the best epoch
            train_stats.update_global(epoch)

        train_stats.export(output_folder)  # update metrics data on disk
        train_stats.plot_all_metrics(output_folder)

    train_stats.export(output_folder)  # update metrics data on disk
    train_stats.plot_all_metrics(output_folder)
    if args.save_model:
        import os
        torch.save(best_model, os.path.join(output_folder, "model.pt"))


if __name__ == '__main__':
    main()
