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

import utils
import lcl_models
import metadata


class MNIST_subset(torch.utils.data.Dataset):

    def __init__(self, src_data, num_classes):
        self.imgs = list()
        self.labels = list()
        class_list = list(range(num_classes))
        for i in range(len(src_data)):
            data, target = src_data.__getitem__(i)
            if target in class_list:
                self.labels.append(target)
                self.imgs.append(data)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.imgs[item], self.labels[item]



class Net(nn.Module):
    def __init__(self, dim, num_classes=10):
        super(Net, self).__init__()

        # ----------
        # The architecture
        # ----------
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, dim)
        self.fc3 = nn.Linear(dim, num_classes)
        self.gmm_layer = lcl_models.axis_aligned_gmm_cmm_layer(dim, num_classes)

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

        resp_gmm, resp_cmm, cluster_dist = self.gmm_layer(x)
        return resp_gmm, resp_cmm, cluster_dist, y


def train(args, model, device, train_loader, optimizer, epoch, train_stats):
    model.train()

    # Setup loss criteria
    cluster_criterion = torch.nn.MSELoss()
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.NLLLoss()
    loss_list = list()
    accuracy_list = list()

    L = 0.5#1.0
    # grad_scale_dict = dict()

    # tmp_stats = metadata.TrainingStats()
    #
    # try:
    nan_count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        target_size = target.size()


        # l = torch.rand(target.shape).to(data.device)
        # l_idx = l > 0.5

        resp_gmm, resp_cmm, cluster_dist, logits = model(data)

        cluster_tgt = torch.zeros_like(cluster_dist)
        cluster_loss = cluster_criterion(cluster_dist, cluster_tgt)

        # output = (resp_gmm + resp_cmm) / 2.0
        # output = resp_gmm  # dropout ish version of combining gmm and cmm
        # output[l_idx, :] = resp_cmm[l_idx, :]
        # output = (l * resp_gmm) + ((1.0 - l) * resp_cmm)

        output = resp_cmm
        linear_loss = criterion(logits, target)

        batch_loss_gmm = criterion(resp_gmm, target)
        batch_loss_cmm = criterion(resp_cmm, target)

        # batch_loss = batch_loss_gmm
        # batch_loss[l_idx] = batch_loss_cmm[l_idx]
        # batch_loss = torch.mean(batch_loss)
        # batch_loss = (l * batch_loss_gmm) + ((1.0 - l) * batch_loss_cmm)
        # batch_loss = batch_loss_gmm + batch_loss_cmm + cluster_loss
        # batch_loss = batch_loss_cmm + batch_loss_kmeans + cluster_loss
        # batch_loss = batch_loss_gmm + cluster_loss
        batch_loss = batch_loss_cmm + cluster_loss
        # batch_loss = batch_loss_cmm + batch_loss_kmeans + cluster_loss
        # batch_loss =  + cluster_loss

        # batch_loss = cluster_loss
        # batch_loss = criterion(output, target)
        if torch.isnan(batch_loss):
            nan_count += 1
            if nan_count > 0.5 * len(train_loader):
                raise RuntimeError("more than 25% of the batches were nan, terminating")

        batch_loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 50)

        # if batch_idx % 10 == 0:
        #     for name, param in model.named_parameters():
        #         a = float(torch.max(torch.abs(param.grad)))
        #         tmp_stats.add(batch_idx, name, a)
        #         # if name == 'gmm_layer.L':
        #         #     b = param.grad.detach().cpu().numpy()
        #         #     print("here")

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

            # for k in grad_scale_dict.keys():
            #     print("    {} |grad| = \n    {}".format(k, grad_scale_dict[k][-10:]))

            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            # 	epoch, batch_idx * len(data), len(train_loader.dataset),
            # 	100. * batch_idx / len(train_loader), batch_loss.item()))
            if args.dry_run:
                break

    # except:
    #     tmp_stats.plot_all_metrics('./tmp-plots', all_one_figure=True)
    #     raise
    #
    # tmp_stats.plot_all_metrics('./tmp-plots', all_one_figure=True)
    # exit(1)
    train_stats.add(epoch, 'train_loss', np.nanmean(loss_list))
    train_stats.add(epoch, 'train_accuracy', np.nanmean(accuracy_list))


def test(model, device, test_loader, epoch, train_stats):
    model.eval()
    cluster_criterion = torch.nn.MSELoss()
    criterion = torch.nn.CrossEntropyLoss()
    loss_list = list()
    loss_gmm_list = list()
    accuracy_gmm_list = list()
    loss_cmm_list = list()
    accuracy_cmm_list = list()
    loss_cluster_list = list()
    loss_linear_list = list()
    accuracy_linear_list = list()

    # model.set_lambda(1.0)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # output = model(data)
            resp_gmm, resp_cmm, cluster_dist, logits = model(data)

            vals = torch.nn.functional.softmax(resp_gmm, dim=-1)
            vals, _ = torch.max(vals, dim=-1)
            vals = vals.detach().cpu().numpy()
            print('max gmm prob = {}'.format(np.max(vals)))

            vals = torch.nn.functional.softmax(resp_cmm, dim=-1)
            vals, _ = torch.max(vals, dim=-1)
            vals = vals.detach().cpu().numpy()
            print('max cmm prob = {}'.format(np.max(vals)))

            # from matplotlib import pyplot as plt
            # plt.figure(figsize=(4, 4))
            # plt.hist(vals, bins=25, label='max_resp')
            # plt.yscale("log")
            # plt.title("epoch {}".format(epoch))
            # plt.legend(loc='upper right')
            # plt.savefig('max-resp-hist-epoch{:03d}.png'.format(epoch))
            # plt.close()

            cluster_loss = cluster_criterion(cluster_dist, torch.zeros_like(cluster_dist))
            loss_cluster_list.append(cluster_loss.item())

            linear_loss = criterion(logits, target)

            # undo log space transform which make CE works well
            # output = torch.log((torch.exp(resp_gmm) + torch.exp(resp_cmm)) / 2.0)
            batch_loss_gmm = criterion(resp_gmm, target)
            batch_loss_cmm = criterion(resp_cmm, target)

            loss = batch_loss_cmm + cluster_loss

            # loss = (batch_loss_gmm + batch_loss_cmm) / 2.0
            # loss = criterion(output, target)
            loss_list.append(loss.item())
            loss_gmm_list.append(batch_loss_gmm.item())
            loss_cmm_list.append(batch_loss_cmm.item())
            loss_linear_list.append(linear_loss.item())

            # acc = torch.argmax(output, dim=-1) == target
            acc_gmm = torch.argmax(resp_gmm, dim=-1) == target
            acc_cmm = torch.argmax(resp_cmm, dim=-1) == target
            acc_linear = torch.argmax(logits, dim=-1) == target
            accuracy_gmm_list.append(torch.mean(acc_gmm, dtype=torch.float32).item())
            accuracy_cmm_list.append(torch.mean(acc_cmm, dtype=torch.float32).item())
            accuracy_linear_list.append(torch.mean(acc_linear, dtype=torch.float32).item())
            # accuracy_list.append(torch.mean(acc, dtype=torch.float32).item())

    test_loss = np.nanmean(loss_list)
    avg_test_loss = test_loss
    # test_acc = np.nanmean(accuracy_list)
    train_stats.add(epoch, 'test_loss', test_loss)
    # train_stats.add(epoch, 'test_accuracy', test_acc)
    # print('Test set: Average loss: {:.4f}, Accuracy: {}'.format(test_loss, test_acc))
    print('Test set: Average loss: {:.4f}'.format(test_loss))

    test_loss = np.nanmean(loss_gmm_list)
    test_acc = np.nanmean(accuracy_gmm_list)
    train_stats.add(epoch, 'test_gmm_loss', test_loss)
    train_stats.add(epoch, 'test_gmm_accuracy', test_acc)
    # avg_test_accuracy = test_acc
    print('Test set (GMM): Average loss: {:.4f}, Accuracy: {}'.format(test_loss, test_acc))

    test_loss = np.nanmean(loss_cmm_list)
    test_acc = np.nanmean(accuracy_cmm_list)
    train_stats.add(epoch, 'test_cmm_loss', test_loss)
    train_stats.add(epoch, 'test_cmm_accuracy', test_acc)
    avg_test_accuracy = test_acc
    print('Test set (CMM): Average loss: {:.4f}, Accuracy: {}'.format(test_loss, test_acc))

    test_loss = np.nanmean(loss_linear_list)
    test_acc = np.nanmean(accuracy_linear_list)
    train_stats.add(epoch, 'test_linear_loss', test_loss)
    train_stats.add(epoch, 'test_linear_accuracy', test_acc)
    print('Test set (Linear): Average loss: {:.4f}, Accuracy: {}'.format(test_loss, test_acc))

    test_loss = np.nanmean(loss_cluster_list)
    train_stats.add(epoch, 'test_cluster_loss', test_loss)
    print('Test set (GMM): Cluster loss: {:.4f}'.format(test_loss))

    return avg_test_loss, avg_test_accuracy


def main(args):
    k = 1
    output_folder = './models-20230409/adadelta-model-{:02d}-{}D-{}class-cmm+cluster'.format(k, args.dim, args.num_classes)
    while os.path.exists(output_folder):
        k += 1
        output_folder = './models-20230409/adadelta-model-{:02d}-{}D-{}class-cmm+cluster'.format(k, args.dim, args.num_classes)

    os.makedirs(output_folder)
    import json
    with open(os.path.join(output_folder, 'args.json'), 'w') as fh:
        json.dump(vars(args), fh, ensure_ascii=True, indent=2)

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

    model = Net(args.dim, num_classes=args.num_classes).to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)

    if args.num_classes != 10:
        train_dataset = MNIST_subset(train_dataset, num_classes=args.num_classes)
        test_dataset = MNIST_subset(test_dataset, num_classes=args.num_classes)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_kwargs['batch_size'] = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)


    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    import lr_scheduler
    plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=10, threshold=1e-4, max_num_lr_reductions=1)

    train_stats = metadata.TrainingStats()
    epoch = -1
    MAX_EPOCHS = 2000
    best_model = copy.deepcopy(model)


    while not plateau_scheduler.is_done() and epoch < MAX_EPOCHS:
        epoch += 1
        train(args, model, device, train_loader, optimizer, epoch, train_stats)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        test_loss, test_acc = test(model, device, test_loader, epoch, train_stats)
        # plateau_scheduler.step(test_loss)
        plateau_scheduler.step(test_acc)

        if plateau_scheduler.is_equiv_to_best_epoch:
            # print('Updating best model with epoch: {} loss: {}'.format(epoch, test_loss))
            print('Updating best model with epoch: {} accuracy: {}'.format(epoch, test_acc))
            best_model = copy.deepcopy(model)

            # update the global metrics with the best epoch
            train_stats.update_global(epoch)

        train_stats.export(output_folder)  # update metrics data on disk
        train_stats.plot_all_metrics(output_folder)

    train_stats.export(output_folder)  # update metrics data on disk
    train_stats.plot_all_metrics(output_folder)

    torch.save(best_model, os.path.join(output_folder, "model.pt"))
    torch.save(best_model.state_dict(), os.path.join(output_folder, "model-state-dict.pt"))

    # build gif, and remove tmp files
    fns = [fn for fn in os.listdir('./') if fn.startswith('feature_space_')]
    fns.sort()
    if len(fns) > 0:
        fps = 2
        if len(fns) > 50:
            fps = 4
        if len(fns) > 100:
            fps = 8
        with imageio.get_writer(os.path.join(output_folder, 'feature_space.gif'), mode='I', fps=fps) as writer:
            for filename in fns:
                image = imageio.imread(filename)
                writer.append_data(image)
        for fn in fns:
            os.remove(fn)

    # build gif, and remove tmp files
    fns = [fn for fn in os.listdir('./') if fn.startswith('D_space_')]
    fns.sort()
    if len(fns) > 0:
        fps = 4
        if len(fns) > 50:
            fps = 8
        if len(fns) > 100:
            fps = 16
        with imageio.get_writer(os.path.join(output_folder, 'D_space.gif'), mode='I', fps=fps) as writer:
            for filename in fns:
                image = imageio.imread(filename)
                writer.append_data(image)
        for fn in fns:
            os.remove(fn)


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--dim', type=int, default=2, metavar='N',
                        help='dimensionality of the gmm')
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
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

    main(args=args)



