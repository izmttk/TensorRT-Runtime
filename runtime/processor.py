from collections import OrderedDict
from typing import Dict, OrderedDict, List, Union
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np

from .utils import get_input_tensor_names, get_output_tensor_names
from .allocator import Allocator

import pycuda.autoinit

class OutputAllocator(trt.IOutputAllocator):
    def __init__(self):
        # print("[MyOutputAllocator::__init__]")
        super().__init__()
        self.buffers = {}
        self.shapes = {}

    def reallocate_output(self, tensor_name: str, memory: int, size: int, alignment: int) -> int:
        # print("[MyOutputAllocator::reallocate_output] TensorName=%s, Memory=%s, Size=%d, Alignment=%d" % (tensor_name, memory, size, alignment))
        if tensor_name in self.buffers:
            self.buffers[tensor_name].free()
        
        address = cuda.mem_alloc(size)
        self.buffers[tensor_name] = address
        return int(address)
        
    def notify_shape(self, tensor_name: str, shape: trt.Dims):
        # print("[MyOutputAllocator::notify_shape] TensorName=%s, Shape=%s" % (tensor_name, shape))
        self.shapes[tensor_name] = tuple(shape)

class ProcessorV3:
    def __init__(self, engine: trt.ICudaEngine, pre_alloc: bool = False):
        self.engine = engine
        self.pre_alloc = pre_alloc
        self.output_allocator = OutputAllocator()
        # create execution context
        self.context = engine.create_execution_context()
        # get input and output tensor names
        self.input_tensor_names = get_input_tensor_names(engine)
        self.output_tensor_names = get_output_tensor_names(engine)
        # create stream
        self.stream = cuda.Stream()
        # Create a CUDA events
        self.start_event = cuda.Event()
        self.end_event = cuda.Event()
        
    def get_last_inference_time(self):
        return self.start_event.time_till(self.end_event)
        
    def inference(self, inputs: Union[Dict[str, np.ndarray], List[np.ndarray], np.ndarray]) -> OrderedDict[str, np.ndarray]:
        """
        inference process:
        1. create execution context
        2. set input shapes
        3. allocate memory
        4. copy input data to device
        5. run inference on device
        6. copy output data to host and reshape
        """
        # set input shapes, the output shapes are inferred automatically

        if isinstance(inputs, np.ndarray):
            inputs = [inputs]
        if isinstance(inputs, dict):
            inputs = [inp if name in self.input_tensor_names else None for (name, inp) in inputs.items()]
        if isinstance(inputs, list):
            for name, arr in zip(self.input_tensor_names, inputs):
                self.context.set_input_shape(name, arr.shape)

        buffers_host = []
        buffers_device = []
        # copy input data to device
        for name, arr in zip(self.input_tensor_names, inputs):
            host = cuda.pagelocked_empty(arr.shape, dtype=trt.nptype(self.engine.get_tensor_dtype(name)))
            device = cuda.mem_alloc(arr.nbytes)
            
            host[:] = arr
            cuda.memcpy_htod_async(device, host, self.stream)
            buffers_host.append(host)
            buffers_device.append(device)
        # set input tensor address
        for name, buffer in zip(self.input_tensor_names, buffers_device):
            self.context.set_tensor_address(name, int(buffer))
        # set output tensor allocator
        for name in self.output_tensor_names:
            self.context.set_tensor_address(name, 0) # set nullptr
            self.context.set_output_allocator(name, self.output_allocator)
        # The do_inference function will return a list of outputs
        
        # Record the start event
        self.start_event.record(self.stream)
        # Run inference.
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        # Record the end event
        self.end_event.record(self.stream)

        # self.memory.copy_to_host()
        
        output_buffers = OrderedDict()
        for name in self.output_tensor_names:
            arr = cuda.pagelocked_empty(self.output_allocator.shapes[name], dtype=trt.nptype(self.engine.get_tensor_dtype(name)))
            cuda.memcpy_dtoh_async(arr, self.output_allocator.buffers[name], stream=self.stream)
            output_buffers[name] = arr
        
        # Synchronize the stream
        self.stream.synchronize()
        
        return output_buffers



class Processor:
    def __init__(self, engine: trt.ICudaEngine, pre_alloc: bool = False):
        self.engine = engine
        self.pre_alloc = pre_alloc
        # create execution context
        self.context = engine.create_execution_context()
        # get input and output tensor names
        self.input_tensor_names = get_input_tensor_names(engine)
        self.output_tensor_names = get_output_tensor_names(engine)
        self.memory = None
        if pre_alloc:
            for name in self.input_tensor_names:
                max_shape = self.get_max_input_shape(name)
                self.context.set_input_shape(name, max_shape)
            self.memory = Allocator(self.context)
        # create stream
        self.stream = cuda.Stream()
        # Create a CUDA events
        self.start_event = cuda.Event()
        self.end_event = cuda.Event()
        
    def __del__(self):
        if self.memory is not None:
            self.memory.free()
            self.memory = None
            
    def get_last_inference_time(self):
        return self.start_event.time_till(self.end_event)
    
    def get_max_input_shape(self, tensor_name: str) -> tuple:
        profile_shapes = self.engine.get_tensor_profile_shape(tensor_name, self.context.active_optimization_profile)
        max_shape = None
        for shape in profile_shapes:
            if max_shape is None:
                max_shape = shape
            elif trt.volume(shape) > trt.volume(max_shape):
                max_shape = shape
        return tuple(max_shape)
        
    def inference(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """
        inference process:
        1. create execution context
        2. set input shapes
        3. allocate memory
        4. copy input data to device
        5. run inference on device
        6. copy output data to host and reshape
        """

        # set input shapes, the output shapes are inferred automatically
        for name, arr in zip(self.input_tensor_names, inputs):
            self.context.set_input_shape(name, arr.shape)

        if not self.pre_alloc:
            # allocate memory for inputs and outputs
            self.memory = Allocator(self.context)
        address_list = self.memory.get_address_list()

        # set input data
        for name, arr in zip(self.input_tensor_names, inputs):
            self.memory.set_binding(name, arr.ravel())
        self.memory.copy_to_device()

        # The do_inference function will return a list of outputs
        
        # Record the start event
        self.start_event.record(self.stream)
        # Run inference.
        self.context.execute_async_v2(bindings=address_list, stream_handle=self.stream.handle)
        # Record the end event
        self.end_event.record(self.stream)
        # Synchronize the stream
        self.stream.synchronize()

        self.memory.copy_to_host()
        
        output_buffers = OrderedDict()
        for name in self.output_tensor_names:
            arr = self.memory.get_binding(name).copy()
            output_shape = tuple(self.context.get_tensor_shape(name))
            arr = arr.reshape(-1)[:np.prod(output_shape)].reshape(output_shape)
            output_buffers[name] = arr
        
        if not self.pre_alloc:
            self.memory.free()
            self.memory = None
        
        return output_buffers
