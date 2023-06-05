from collections import OrderedDict
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np

class HostDeviceMemory:
    """Pair of host and device memory, where the host memory is wrapped in a numpy array"""
    def __init__(self, size: int, dtype: np.dtype):
        nbytes = size * dtype.itemsize
        self._host = cuda.pagelocked_empty(size, dtype)
        self._device = cuda.mem_alloc(nbytes)
        self._nbytes = nbytes

    @property
    def host(self):
        return self._host

    @host.setter
    def host(self, arr: np.ndarray):
        if arr.size > self.host.size:
            raise ValueError(f"Tried to fit an array of size {arr.size} into host memory of size {self.host.size}")
        np.copyto(self.host[:arr.size], arr.flat, casting='safe')

    @property
    def device(self):
        return int(self._device)

    @property
    def nbytes(self):
        return self._nbytes

    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}\nSize:\n{self.nbytes}\n"

    def __repr__(self):
        return self.__str__()

    def free(self):
        self._device.free()
        self._host.base.free()


class Allocator:
    def __init__(self, context: trt.IExecutionContext):
        self.context = context
        self.engine = context.engine 
        self.bindings = OrderedDict()
        self.stream = cuda.Stream()
        self.allocate()

    def allocate(self):
        for binding in self.engine:
            tensor_shape = self.context.get_tensor_shape(binding)
            tensor_dtype = self.engine.get_tensor_dtype(binding)
            memory = HostDeviceMemory(trt.volume(tensor_shape), np.dtype(trt.nptype(tensor_dtype)))
            self.bindings[binding] = memory
            
    def free(self):
        for binding in self.bindings.values():
            binding.free()
            
    def copy_to_host(self):
        for binding in self.bindings.values():
            cuda.memcpy_dtoh_async(binding.host, binding.device, self.stream)
        self.stream.synchronize()
            
    def copy_to_device(self):
        for binding in self.bindings.values():
            cuda.memcpy_htod_async(binding.device, binding.host, self.stream)
        self.stream.synchronize()
            
    def get_binding(self, tensor_name: str, copy_to_host = False):
        binding = self.bindings[tensor_name]
        if copy_to_host:
            # copy to host
            cuda.memcpy_dtoh_async(binding.host, binding.device, self.stream)
            self.stream.synchronize()
        return binding.host

    def set_binding(self, tensor_name: str, arr: np.ndarray, copy_to_device = False):
        binding = self.bindings[tensor_name]
        binding.host = arr
        if copy_to_device:
            # copy to device
            cuda.memcpy_htod_async(binding.device, binding.host, self.stream)
            self.stream.synchronize()
    
    def get_address(self, tensor_name: str):
        return int(self.bindings[tensor_name].device)
    
    def get_address_list(self):
        return [int(binding.device) for binding in self.bindings.values()]
    
    
# def allocate_context_memory(context: trt.IExecutionContext) -> BindingMemoryAllocation:
#     return BindingMemoryAllocation(context)

# def free_context_memory(alloc: BindingMemoryAllocation):
#     alloc.free()