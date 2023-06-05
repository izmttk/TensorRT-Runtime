import tensorrt as trt
import pycuda.driver as cuda
from . import logger

def get_input_tensor_names(engine: trt.ICudaEngine):
    input_tensor_names = []
    for binding in engine:
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            input_tensor_names.append(binding)
    return input_tensor_names

def get_output_tensor_names(engine: trt.ICudaEngine):
    output_tensor_names = []
    for binding in engine:
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.OUTPUT:
            output_tensor_names.append(binding)
    return output_tensor_names


def save_engine(engine, path):
    logger.info(f'Saving engine to file {path}')
    with open(path, 'wb') as f:
        f.write(engine)
    logger.info('Completed saving engine')

def load_engine(path):
    logger.info(f'Loading engine from file {path}')
    runtime = trt.Runtime(logger.TRT_LOGGER)
    with open(path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    logger.info('Completed loading engine')
    return engine