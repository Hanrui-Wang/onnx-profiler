# ONNX Profiler

This library is used to profile the ONNX or PyTorch models

## Installment

Using GitHub: `pip install --upgrade git+https://github.com/zhijian-liu/onnx-profiler.git`

## Usage

```
import numpy as np
import torch
from torchvision.models import *
from onnxp import *

model = resnet18()
inputs = torch.randn(1, 3, 224, 224)

mults = torch_profile(model, inputs, profiler=OperationsProfiler, reduction=np.sum)
params = torch_profile(model, inputs, profiler=ParametersProfiler, reduction=np.sum)
```

## License

This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.