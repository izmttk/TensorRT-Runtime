import argparse
from pathlib import Path
import sys
import os

import tensorrt as trt

batch_size = 1 # config.batch_size

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def GiB(val):
    return val * 1 << 30

# This function builds an engine from a onnx model.
def build_engine(onnx_file_path, precision = 'fp32', dynamic_shapes = None):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""

    EXPLICIT_BATCH_FLAG = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH_FLAG)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse model file
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Loading ONNX file from path {onnx_file_path}...')
    with open(onnx_file_path, 'rb') as model:
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Beginning ONNX file parsing')
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                TRT_LOGGER.log(TRT_LOGGER.ERROR, parser.get_error(error))
            raise ValueError('Failed to parse the ONNX file.')
    TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed parsing of ONNX file')
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Input number: {network.num_inputs}')
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Output number: {network.num_outputs}')
    
    
    if dynamic_shapes is not None:
        # set optimization profile for dynamic shape
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            input = network.get_input(i)
            min_shape = dynamic_shapes['min_shape']
            opt_shape = dynamic_shapes['opt_shape']
            max_shape = dynamic_shapes['max_shape']
            profile.set_shape(input.name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

    # We set the builder batch size to be the same as the calibrator's, as we use the same batches
    # during inference. Note that this is not required in general, and inference batch size is
    # independent of calibration batch size.
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, GiB(1)) # 1G
    
    if precision == 'fp32':
        pass
    elif precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
    else:
        raise ValueError('precision must be one of fp32, fp16, or int8')

    # Build engine.
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Building an engine from file {onnx_file_path}; this may take a while...')
    serialized_engine = builder.build_serialized_network(network, config)
    TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed creating Engine')
    return serialized_engine

def save_engine(engine, path):
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Saving engine to file {path}')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(engine)
    TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed saving engine')

def load_engine(path):
    TRT_LOGGER.log(TRT_LOGGER.WARNING, f'Loading engine from file {path}')
    runtime = trt.Runtime(TRT_LOGGER)
    with open(path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed loading engine')
    return engine
    
def run(
        onnx,  # onnx file path
        output='exports', # output file path
        int8=False,  # batch size
        dynamic=False,  # dynamic input shape
        min_shape=[1, 3, 192, 192],  # min input size
        opt_shape=[1, 3, 640, 640],  # optimal input size
        max_shape=[1, 3, 960, 960],  # max input size
        verbose=False,  # verbose output
):
    if dynamic:
        dynamic_shapes = {
            'min_shape': min_shape,
            'opt_shape': opt_shape,
            'max_shape': max_shape
        }
    else:
        dynamic_shapes = None
    engine = build_engine(onnx, 'int8' if int8 else 'fp32', dynamic_shapes)
    save_engine(engine, output)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, help='onnx file path')
    parser.add_argument('--output', type=str, default='exports', help='output engine file path')
    parser.add_argument('--int8', action='store_true', help='int8 precision')
    parser.add_argument('--dynamic', action='store_true', help='dynamic input shape')
    parser.add_argument('--min-shape', nargs='+', type=int, help='min input size')
    parser.add_argument('--opt-shape', '--shape', nargs='+', type=int, help='optimal input size')
    parser.add_argument('--max-shape', nargs='+', type=int, help='max input size')
    opt = parser.parse_args()
    return opt


def main(opt):
    if (opt.max_shape is None or opt.min_shape is None) and opt.opt_shape is not None:
        opt.max_shape = opt.min_shape = opt.opt_shape
    run(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
    
#     dynamic_shapes = {
#         'min_shape': [1, 3, 192, 192],
#         'opt_shape': [max(1, batch_size // 2), 3, 640, 640],
#         'max_shape': [batch_size, 3, 960, 960]
#     }
    
#     # engine = build_engine('./exports/model.onnx', 'fp32', dynamic_shapes)
#     # save_engine(engine, './exports/model.trt')

#     engine = build_engine('./exports/model_quantized.onnx', 'int8', dynamic_shapes)
#     save_engine(engine, './exports/model_quantized.trt')
    