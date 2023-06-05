import argparse
from pathlib import Path
import sys
import os
import onnx

import torch
import torchvision
from torchvision.transforms import functional as F, transforms as T
from tqdm import tqdm

from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib

# from quantization.utils.plots import plot_weights_histogram, plot_activation_histogarm

def collect_stats(model, dataloader, max_batch):
    """Feed data to the network and collect statistic"""
    device = next(model.parameters()).device  # get model device
    with torch.no_grad():
        # Enable calibrators
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()
        # Feed data to the network for collecting stats

        for i, (images, targets) in enumerate(dataloader):
            images = list(image.to(device) for image in images)
            model(images)
            if max_batch is not None and i >= max_batch:
                break

        # Disable calibrators
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()

def compute_amax(model, **kwargs):
    """Load calib result"""
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax()
                    else:
                        module.load_calib_amax(**kwargs)

def export_onnx(model, im, onnx_file, dynamic, device='cpu'):
    # ONNX export
    import onnx

    print(f'\nStarting export with onnx {onnx.__version__}...')

    if dynamic:
        # modify this for your model
        dynamic = {
            'images': {1: 'height', 2: 'width'}, # shape(1,3,640,640)
            'boxes': {0: 'anchors'},
            'scores': {0: 'anchors'},
            'labels': {0: 'anchors'}
        }
    
    path = Path(onnx_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model.cpu() if dynamic else model.to(device),  # --dynamic only compatible with cpu
        ([im.cpu()],) if dynamic else ([im.to(device)],),
        onnx_file,
        verbose=False,
        do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        input_names=['images'],
        output_names=['boxes', 'scores', 'labels'],
        dynamic_axes=dynamic or None)

    # Checks
    model_onnx = onnx.load(onnx_file)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    onnx.save(model_onnx, onnx_file)
    
    # Simplify
    # import onnxsim
    # model_onnx, check = onnxsim.simplify(model_onnx)
    # assert check, 'Onnx Simplify failed'
    # onnx.save(model_onnx, onnx_file)

def run(
        data_root, # dataset root directory
        data_ann,  # annotation json file
        weight=None,  # model.pt path(s)
        imgsz=640,  # inference size (pixels)
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        output='model.onnx',  # save to project/name
        quant=False,  # quantize
        calib='max',  # activation calibration method
        calib_percentile=99.99,  # activation calibration percentile
        calib_batch_size=32,  # max batch size for calibration
        max_batch=2,  # max batch size for calibration
        dynamic=False,  # dynamic input/output
):
    device = torch.device(device)

    if quant:
        
        # An experimental static switch for using pytorch's native fake quantization
        # Primary usage is to export to ONNX
        quant_nn.TensorQuantizer.use_fb_fake_quant = True

        if calib == 'max':
            calib_method = 'max'
        elif calib in ['percentile', 'mse', 'entropy']:
            calib_method = 'histogram'
        else:
            raise NotImplementedError(f'Unsupported calibration method: {calib}')

        # Set default QuantDescriptor to use histogram based calibration for activation
        quant_desc_input = QuantDescriptor(num_bits=8, calib_method=calib_method)
        
        # Calibrator histogram collection only supports per tensor scaling
        quant_desc_weight = QuantDescriptor(num_bits=8, calib_method='max', axis=(0))
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantConv2d.set_default_quant_desc_weight(quant_desc_weight)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLinear.set_default_quant_desc_weight(quant_desc_weight)
        quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
        # Initialize quantized modules
        quant_modules.initialize()
    
    model = torch.load(weight, map_location='cpu')
    # if only save model state dict
    # state_dict = torch.load(weight, map_location='cpu')
    # model.load_state_dict(state_dict['model'], strict=False)
    model = model.to(device).float()  # FP32
    model = model.eval()
    
    # warm up
    im = torch.zeros(3, imgsz, imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection
    for _ in range(2):
        y = model([im])  # dry runs

    if quant:
        # Data
        transform = T.Compose([
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float),
        ])
        
        train_datasets = torchvision.datasets.CocoDetection(
            root=data_root,
            annFile=data_ann,
            transform=transform,
        )

        train_sampler = torch.utils.data.RandomSampler(train_datasets)
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, calib_batch_size, drop_last=True)

        train_loader = torch.utils.data.DataLoader(
            train_datasets,
            batch_sampler=train_batch_sampler,
            collate_fn=lambda batch: tuple(zip(*batch)),
            num_workers=workers
        )
        # print(f'\nCollecting weight and activation stats...')
        # plot_weights_histogram(model, save_path=export_dir)
        # print(f'Weights histogram saved to {export_dir / "weights_histogram.jpg"}')
        # layer_name = 'model.3'
        # plot_activation_histogarm(model, train_loader, layer_name, save_path=export_dir)
        # print(f'Activation histogram saved to {export_dir / f"activation_histogram_{layer_name}.jpg"}')
        print(f'\nCollecting weight and activation stats...')
        collect_stats(model, train_loader, max_batch=max_batch)
        print(f'\nComputing amax...')
        compute_amax(model, method=calib, percentile=calib_percentile if calib == 'percentile' else None)
    
    # print(f'Layer {layer_name} weight amax: {dict(model.named_modules())["model.3.conv"].weight_quantizer.amax}')
    # print(f'Layer {layer_name} input amax: {dict(model.named_modules())["model.3.conv"].input_quantizer.amax}')

    export_onnx(model, im, output, dynamic)
    print(f'\nExport complete')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='datasets/coco2017/train', help='dataset root path')
    parser.add_argument('--data-ann', type=str, default='datasets/coco2017/instances_train2017.json', help='annotation json file path')
    parser.add_argument('--weight', type=str, default='model.pt', help='weight path')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--output', default='model.onnx', help='save to project/name')
    parser.add_argument('--quant', action='store_true', help='quantize model')
    parser.add_argument('--calib', default='max', help='calibration method for activation', choices=['max', 'percentile', 'mse', 'entropy'])
    parser.add_argument('--calib-percentile', type=float, default=99.99, help='calibration percentile for percentile method')
    parser.add_argument('--calib-batch-size', type=int, default=32, help='calibration batch size')
    parser.add_argument('--max-batch', type=int, default=2, help='max batch size for calibration')
    parser.add_argument('--dynamic', action='store_true', help='use dynamic quantization')

    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
