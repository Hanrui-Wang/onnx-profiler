# ONNX Profiler

This tool is used to profile the NN models (currently supporting ONNX and PyTorch models).

## Installation

Using GitHub

```bash
pip install --upgrade git+https://github.com/zhijian-liu/onnx-profiler.git
```

## Getting Started (PyTorch)

Before profiling, you should first define your PyTorch model and a (dummy) input which the model takes:

```python
from torchvision.models import *
import torch

model = resnet18()
inputs = torch.randn(1, 3, 224, 224)
```

If you want to profile the number of parameters in your model,

```python
from onnxp import *
import numpy as np

mults = torch_profile(model, inputs, profiler=OperationsProfiler, reduction=np.sum, verbose=True)
```

Here, we explain a little bit more on the parameters of `torch_profile`:

* `verbose` is set to `True` to display the number of multiplications in each layer; alternatively, you can set to `False` if you do not want any intermediate outputs.
* `reduction` is set to `np.sum` to sum up the computations in all layers; alternatively, you can set to `None` to obtain the computations in all layers (as a list).

If you want to profile the number of parameters in your model,

```python
from onnxp import *
import numpy as np

params = torch_profile(model, inputs, profiler=ParametersProfiler, reduction=np.sum, verbose=True)
```

Similarly, if you want to display the output activation sizes of all layers in your model,

```python
from onnxp import *
import numpy as np

torch_profile(model, inputs, profiler=ActivationProfile, verbose=True)
```

## License

This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.