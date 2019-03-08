# ONNX Profiler

This tool is used to profile the NN models (currently supporting ONNX and PyTorch models).

## Installment

Using GitHub

```bash
pip install --upgrade git+https://github.com/zhijian-liu/onnx-profiler.git
```

## Getting Started (PyTorch)

Before profiling, you should first define the PyTorch model and a (dummy) input the model takes:

```python
import torch
from torchvision.models import *

model = resnet18()
inputs = torch.randn(1, 3, 224, 224)
```

As for the number of multiplications, you might use the following command (where `verbose` is set to `True` to display the number of multiplications in each layer, and `reduction` is set to `np.sum` to sum up the computations in all layers):

```python
import numpy as np
from onnxp import *

mults = torch_profile(model, inputs, profiler=OperationsProfiler, reduction=np.sum, verbose=True)
```

For the number of parameters, you might use the following command:

```python
import numpy as np
from onnxp import *

params = torch_profile(model, inputs, profiler=ParametersProfiler, reduction=np.sum, verbose=True)
```

You might also use the following command to display the output activation sizes for all layers:

```python
import numpy as np
from onnxp import *

torch_profile(model, inputs, profiler=ActivationProfile, verbose=True)
```

## License

This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.