import os
import shutil
import numpy as np
import subprocess
import torch
import logging
import inspect


# adapted from https://github.com/karpathy/nanoGPT/blob/master/model.py#L270
def configure_optimizer(model, weight_decay, learning_rate, method='sgd', nesterov=True):
    if weight_decay is None:
        weight_decay = 0.0

    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logging.info("num decayed parameter tensors: {}, with {} parameters".format(len(decay_params), num_decay_params))
    logging.info("num non-decayed parameter tensors: {}, with {} parameters".format(len(nodecay_params), num_nodecay_params))

    if method == 'sgd':
        optimizer = torch.optim.SGD(optim_groups, lr=learning_rate, momentum=0.9, nesterov=nesterov)
        logging.info("Using SGD")
    elif method == 'adamw':
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        extra_args = dict(fused=True) if fused_available else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, **extra_args)
        logging.info("Using fused AdamW: {}".format(fused_available))
    else:
        raise RuntimeError("Invalid optimizer: {}".format(method))
    return optimizer


def validate_output_directory(args):
    if args.debug:
        # if we are in debug mode delete any existing output data.
        if os.path.exists(args.output_dirpath):
            shutil.rmtree(args.output_dirpath)
    else:
        if os.path.exists(args.output_dirpath):
            # if we are not in debug mode, preserve all output data
            raise RuntimeError("Output dirpath {} exists, exiting.".format(args.output_dirpath))

    os.makedirs(args.output_dirpath)


def is_ide_debug():
    # check if IDE is in debug mode, and set num parallel worker to 0
    import sys
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        print("Detected IDE debug mode, force enabling debug mode and setting number of workers to 0")
        return True
    return False

def check_for_ide_debug_mode(args):
    if is_ide_debug():
        args.num_workers = 0
        args.debug = True


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_used_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    command = "nvidia-smi --query-gpu=memory.total --format=csv"
    memory_total_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_used_info = [int(x.split()[0]) for i, x in enumerate(memory_used_info)]
    memory_total_info = [int(x.split()[0]) for i, x in enumerate(memory_total_info)]
    memory_used_percent = np.asarray(memory_used_info) / np.asarray(memory_total_info)
    return memory_used_percent, memory_total_info


def compute_class_prevalance(dataloader):
    label_list = list()
    with torch.no_grad():
        for batch_idx, tensor_dict in enumerate(dataloader):
            #inputs = tensor_dict[0].cuda()
            labels = tensor_dict[1].cuda()

            label_list.append(labels.detach().cpu().numpy())

    label_list = np.concatenate(label_list).reshape(-1)
    unique_labels = np.unique(label_list)
    N = len(label_list)
    class_preval = {}
    for i in range(len(unique_labels)):
        c = unique_labels[i]
        count = np.sum(label_list == c)
        class_preval[c] = count/N

    return class_preval



def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])
    
    
def multiconcat_numpy(L):

    # is it 1D
    is1d = (len(L[0].shape)==1)

    nRows = 0
    nCols = 1 if is1d else L[0].shape[1]
    dtype = L[0].dtype
    #dtype = np.float32
    
    # count the rows, and double check the columns
    for x in L:
        rows = x.shape[0]
        cols = 1 if is1d else x.shape[1]
        nRows += rows
        if (cols!=nCols):
            raise RuntimeError("ERROR: concat_numpy cols do not match:  cols %d nCols %d" % (cols, nCols) )
    logging.info("multiconcat_numpy nRows %d nCols %d" % (nRows,nCols))
    
    # concatenate all of the numpy arrays
    if is1d:
        A = np.zeros((nRows), dtype=dtype)
    else:
        A = np.zeros((nRows,nCols), dtype=dtype)
    sidx=0
    eidx=0
    for x in L:
        sidx = eidx
        eidx += x.shape[0]
        A[sidx:eidx] = x
    
    return A
    
