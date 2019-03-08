# ONNX Profiler

This tool is used to profile the NN models (supporting ONNX and PyTorch models).

## Installment

Using GitHub

`pip install --upgrade git+https://github.com/zhijian-liu/onnx-profiler.git`

## Usage

```python
import torch
from torchvision.models import *

model = resnet18()
inputs = torch.randn(1, 3, 224, 224)
```

If you want to profile this model,

```python
from onnxp import *
import numpy as np

mults = torch_profile(model, inputs, profiler=OperationsProfiler, reduction=np.sum, verbose=True)
```

## License

This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.