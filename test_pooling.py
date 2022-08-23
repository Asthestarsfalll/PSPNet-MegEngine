from models.pooling import AdaptiveAvgPooling2D
from torch.nn import AdaptiveAvgPool2d
import numpy as np
import torch
import megengine as mge

def test_func(mge_out, torch_out):
    mge_out = mge_out.numpy()
    torch_out = torch_out.detach().numpy()
    result = np.isclose(mge_out, torch_out, rtol=1e-3)
    ratio = np.mean(result)
    allclose = np.all(result) > 0
    abs_err = np.mean(np.abs(mge_out - torch_out))
    std_err = np.std(np.abs(mge_out - torch_out))
    return ratio, allclose, abs_err, std_err


m = AdaptiveAvgPooling2D(7)
t = AdaptiveAvgPool2d(output_size=7)

inp = np.random.randn(2, 3, 223, 224)
m_inp = mge.tensor(inp, dtype=np.float32)
t_inp = torch.tensor(inp, dtype=torch.float32)
t_out = t(t_inp)
m_out = m(m_inp)
ratio, allclose, abs_err, std_err = test_func(m_out, t_out)
print(f"Result: {allclose}, {ratio*100 : .4f}% elements is close enough\n which absolute error is  {abs_err} and absolute std is {std_err}")
