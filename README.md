# PSPNet-MegEngine

The MegEngine implementation of PSPNet.

**Note:**

This implementation may be much slower than the torch implementation, due to there is a big gap in the `AdaptiveAvgPool` API between megengine and torch.

In MegEngine, `AdaptiveAvgPool` is adpted from `AvgPool` by automatically infer `kernel_size` and `stride`. However, torch's implementation of `AdaptiveAvgPool` are highly diferent which uses diferent `kernel_size`  and `stride` when sliding window. You can get more details from [here](https://github.com/pytorch/pytorch/blob/877c96cddfebee00385307f9e1b1f3b4ec72bfdc/aten/src/ATen/native/AdaptiveAveragePooling.cpp#L12-L72).

Check the `class AdaptiveAvgPooling2D()` in models.pooling to help you understand how torch's implementation do.

## Usage

### Install Dependencies

```bash
pip install -r requirements.txt
```

If you don't want to compare the ouput error between the MegEngine implementation and PyTorch one, just ignore requirements.txt and install MegEngine from the command line:

```bash
python3 -m pip install --upgrade pip 
python3 -m pip install megengine -f https://megengine.org.cn/whl/mge.html
```

### Convert weights

Convert trained weights from torch to megengine, the converted weights will be saved in ./pretained/ , you need to specify the convert model architecture and path to checkpoint offered by [official repo](https://github.com/AlfredXiangWu/LightCNN#evaluation).

```bash
python convert_weights.py -m resnet50 -c /path/to/ckpt
```

### Compare

Use `python compare.py` .

By default, the compare script will convert the torch state_dict to the format that megengine need.

If you want to compare the error by checkpoints, you neet load them manually.

### Load From Hub

Import from megengine.hub:

Way 1:

```python
from megengine import hub

modelhub = hub.import_module(
    repo_info='asthestarsfalll/PSPNet-MegEngine:main', git_host='github.com')

# load pretrained model
pretrained_model = modelhub.pspnet50(pretrained=True)
```

Way 2:

```python
from  megengine import hub

# load pretrained model 
model_name = 'pspnet50'
pretrained_model = hub.load(
    repo_info='asthestarsfalll/PSPNet-MegEngine:main', entry=model_name, git_host='github.com', pretrained=True)
```

For those models which do not have pretrained model online, you need to convert weights mannaly, and load the model without pretrained weights like this:

```python
model = modelhub.pspnet50()
# or
model_name = 'pspnet50'
model = hub.load(
    repo_info='asthestarsfalll/PSPNet-MegEngine:main', entry=model_name, git_host='github.com')
```

## Reference

[The official pytorch implementation of PSPNet](https://github.com/hszhao/semseg)