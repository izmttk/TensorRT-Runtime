# TensorRT Runtime for Python

This is a Python wrapper for the TensorRT runtime API. It is intended to be used with TensorRT.

**Attention**: This repo is for study purposes only. If you want to use it in production, please **proceed at your own risk**. The code has only been tested with TensorRT 8.6.0.

## Installation

You should install the TensorRT first, for more information, please refer to [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).

Then you should install the following dependencies. Dependency versions are not strict, but the code has only been tested with the following versions.

```plain
torch>=1.7.0
pytorch-quantization>=2.1.2
matplotlib>=3.2.2
numpy>=1.24.3
tensorrt=8.6.0
pycuda>=2022.2.2
```

## Usage

### Quantization

Please refer to [pytorch-quantization](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/index.html#) and file [export_onnx.py](./export_onnx.py) for more information. You may need to modify the code to fit your own model.

### Build Engine

```plain
usage: onnx2trt.py [-arguments]

optional arguments:
  -h, --help            show this help message and exit
  --onnx ONNX           onnx file path
  --output OUTPUT       output engine file path
  --int8                int8 precision mode
  --dynamic             dynamic input shape
  --min-shape MIN_SHAPE [MIN_SHAPE ...]
                        min input size
  --opt-shape OPT_SHAPE [OPT_SHAPE ...], --shape OPT_SHAPE [OPT_SHAPE ...]
                        optimal input size
  --max-shape MAX_SHAPE [MAX_SHAPE ...]
                        max input size
```

You can use any other tools, such as [trtexec](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec), to convert your model to a TensorRT engine.

### Inference

```python
from runtime import Processor
from utils import load_engine

# load serialized engine file
engine = load_engine("path/to/engine_file")
# create processor
processor = ProcessorV3(engine, pre_alloc=True)

# you can also use Processor which use TensorRT's v2 api
# processor = Processor(engine, pre_alloc=True)

# prepare input data
input_data = {
    "your_input_name1": # your input data
    "your_input_name2": # your input data
}

# or you can use list, just make sure the order is correct

# input_data = [your_input_data_1, your_input_data_2, ...]

# inference
output_data = processor(input_data)
```

The output data is a dict, you can retrieve the output data by names specified in your onnx model.

## Problems

1. The code use pycuda to access cuda api, but there exists some problems when using multiple threads. This issue might be resolved by using [cuda python](https://nvidia.github.io/cuda-python/)
2. gpu memory using is not strictly checked, it may cause memory leak.
3. The code havn't using cuda's stream to parallelize the execution of multiple engines, it may cause performance loss.

Any suggestions are welcome.