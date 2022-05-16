import numpy as np
import torch
import torchvision
import torchvision.transforms


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

# due to pytorch + numpy bug
# https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_cifar10(args, lcl_fldr='./data'):
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])

    train_dataset = torchvision.datasets.CIFAR10(lcl_fldr, train=True, download=True, transform=transform_train)

    # use 90% of the train for train, and 10% for val
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=worker_init_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=worker_init_fn)



    test_dataset = torchvision.datasets.CIFAR10(lcl_fldr, train=False, download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=worker_init_fn)

    return train_loader, val_loader, test_loader
