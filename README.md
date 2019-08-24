# ONNX Profiler

This tool is used to profile the NN models (currently supporting ONNX and PyTorch models).

## Installation

We recommend you to install the latest version of this package from GitHub:

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

If you want to profile the number of multiplications in your model,

```python
from onnxp import *
import numpy as np

mults = torch_profile(model, inputs, profiler=OperationsProfiler, reduction=np.sum, verbose=True)
```

Here, we explain a little bit more about the arguments of `torch_profile`:

* `reduction` is set to `np.sum` to sum up the computations in all layers. If you want to keep the computations in all layers as a list, you can then set this argument to `None` (which is the default value).
* `verbose` is set to `True` to display the number of multiplications in each layer; alternatively, if you do not want to display any intermediate output, you can also set it to `False` (which is the default value).

Similarly, if you want to profile the number of parameters in your model,

```python
from onnxp import *
import numpy as np

params = torch_profile(model, inputs, profiler=ParametersProfiler, reduction=np.sum, verbose=True)
```

Further, if you want to display the output activation sizes of all layers in your model,

```python
from onnxp import *
import numpy as np

torch_profile(model, inputs, profiler=ActivationProfiler, verbose=True)
```

## License

This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.
