import time

import megengine as mge
import numpy as np
import torch

from models.pspnet import PSPNet, pspnet50, pspnet101, pspnet152
from models.torch_model import PSPNet as torch_PSPNet
from convert_weights import convert

model_name = "resnet50"

LAYER_MAPPER ={
    "resnet50": 50,
    "resnet101": 101,
    "resnet152": 152,
}

MGE_MAPPER =  {
    "resnet50": pspnet50,
    "resnet101": pspnet101,
    "resnet152": pspnet152,
}

mge_model = MGE_MAPPER[model_name](pretrained=True, classes=19)


torch_model = torch_PSPNet(layers=LAYER_MAPPER[model_name], classes=19)

s = torch_model.state_dict()
m = convert(torch_model, s)
mge_model.load_state_dict(m)

mge_model.eval()
torch_model.eval()

torch_time = meg_time = 0.0

def test_func(mge_out, torch_out):
    result = np.isclose(mge_out, torch_out, rtol=1e-3)
    ratio = np.mean(result)
    allclose = np.all(result) > 0
    abs_err = np.mean(np.abs(mge_out - torch_out))
    std_err = np.std(np.abs(mge_out - torch_out))
    return ratio, allclose, abs_err, std_err


def argmax(logits):
    return np.argmax(logits, axis=1)


for i in range(15):
    results = []
    inp = np.random.randn(2, 3, 225, 225)
    mge_inp = mge.tensor(inp, dtype=np.float32)
    torch_inp = torch.tensor(inp, dtype=torch.float32)

    if torch.cuda.is_available():
        torch_inp = torch_inp.cuda()
        torch_model.cuda()

    st = time.time()
    mge_out = mge_model(mge_inp)
    meg_time += time.time() - st

    st = time.time()
    torch_out = torch_model(torch_inp)
    torch_time += time.time() - st

    if torch.cuda.is_available():
        torch_out = torch_out.detach().cpu().numpy()
    else:
        torch_out = torch_out.detach().numpy()
    mge_out = mge_out.numpy()
    mge_out = argmax(mge_out)
    torch_out = argmax(torch_out)
    ratio, allclose, abs_err, std_err = test_func(mge_out, torch_out)
    results.append(allclose)
    print(f"Result: {allclose}, {ratio*100 : .4f}% elements is close enough\n which absolute error is  {abs_err} and absolute std is {std_err}")

assert all(results), "not aligned"

print(f"meg time: {meg_time}, torch time: {torch_time}")
