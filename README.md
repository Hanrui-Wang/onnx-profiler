# ONNX Profiler

This tool is used to profile the NN models (currently supporting ONNX and PyTorch models).

## Installment

Using GitHub

```bash
pip install --upgrade git+https://github.com/zhijian-liu/onnx-profiler.git
```

## Usage

You first need to define the model

```python
import torch
from torchvision.models import *

model = resnet18()
inputs = torch.randn(1, 3, 224, 224)
```

If you want to profile this model,

```python
import numpy as np
from onnxp import *

mults = torch_profile(model, inputs, profiler=OperationsProfiler, reduction=np.sum, verbose=True)
```

If you want to profile this model,

```python
import numpy as np
from onnxp import *

params = torch_profile(model, inputs, profiler=ParametersProfiler, reduction=np.sum, verbose=True)
```

## License

This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.