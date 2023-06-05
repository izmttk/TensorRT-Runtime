import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn
import numpy as np
matplotlib.rc('font', **{'size': 11})
matplotlib.use('TkAgg')  # for writing to files only

from pathlib import Path

def plot_weights_histogram(model, save_path=None):
    """
    Plots histogram of weights in the model
    Args:
        model: model to plot weights histogram
        save_path: path to save the plot
    """
    fig = plt.figure()
    weights = []
    for name, param in model.named_parameters():
        if 'weight' in name or 'bias' in name:
            weights.append(param.detach().numpy().flatten())
    weights = np.concatenate(weights)
    max_val = np.max(np.abs(weights))
    plt.hist(weights, 1024, range=(-max_val, max_val), log=True, label='weights')
    plt.legend(loc='upper right')
    plt.title('Weights histogram')
    if save_path is not None:
        plt.savefig(Path(save_path) / 'weights_hist.jpg', dpi=200)
    return fig


def plot_activation_histogarm(model, data_loader, layer_name, max_batch = 3, save_path=None):
    """
    Plots histogram of activations in the model
    Args:
        model: model to plot activations histogram
        input: list of input to the model
        save_path: path to save the plot
    """
    fig = plt.figure()
    activations = []
    handles = []
    for name, module in model.named_modules():
        if name == layer_name:
            def hook(module, args, output):
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        activations.append(arg.detach().numpy().flatten())
                    elif isinstance(arg, (list, tuple)):
                        for input in arg:
                            if isinstance(input, torch.Tensor):
                                activations.append(input.detach().numpy().flatten())
            handles.append(module.register_forward_hook(hook))
    for i, (imgs, targets, paths, shapes) in enumerate(data_loader):
        imgs = imgs.float() # uint8 to fp16/32
        imgs /= 255.0  # 0 - 255 to 0.0 - 1.0
        model(imgs)
        if i >= max_batch:
            break
    for handle in handles:
        handle.remove()
    activations = np.concatenate(activations) if len(activations) > 0 else np.array([])
    max_val = np.max(activations) if len(activations) > 0 else 1
    min_val = np.min(activations) if len(activations) > 0 else -1
    plt.hist(activations, 1024, range=(min_val, max_val), log=True, label='activations')
    plt.xlim(min_val, max_val)
    plt.legend(loc='upper right')
    plt.title('Activations histogram')
    if save_path is not None:
        plt.savefig(Path(save_path) / f'activations_hist_{layer_name}.jpg' , dpi=200)
    return fig