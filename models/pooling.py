import megengine.module as M
import megengine.functional as F
from typing import Union, Tuple


class AdaptiveAvgPooling2D(M.Module):
    """ 
        use python to implement AdaptiveAvgPool2D in pytorch
    """

    def __init__(
        self,
        oshp: Union[int, Tuple[int, int]],
    ):
        super(AdaptiveAvgPooling2D, self).__init__()
        if isinstance(oshp, int):
            oshp = (oshp, oshp)
        self.oshp = oshp

    def _cal_kernel_size(self, ishp):
        kh = (ishp[0] + self.oshp[0] - 1) // self.oshp[0]
        kw = (ishp[1] + self.oshp[1] - 1) // self.oshp[1]
        return (kh, kw)

    @staticmethod
    def _zip(*x):
        element_length = len(x[0])
        for i in x:
            assert element_length == len(i)
        out = []
        total_length = len(x)
        for i in range(element_length):
            temp = []
            for j in range(total_length):
                temp.append(x[j][i])
            out.append(temp)
        return out

    def _get_points(self, input_size, kernel_size):
        start_points_h = (F.arange(
            self.oshp[0], dtype='float32') * (input_size[0] / self.oshp[0])).astype('int32')
        end_points_h = F.ceil(((F.arange(
            self.oshp[0], dtype='float32') + 1) * (input_size[0] / self.oshp[0]))).astype('int32')
        start_points_w = (F.arange(
            self.oshp[1], dtype='float32') * input_size[1] / self.oshp[1]).astype('int32')
        end_points_w = F.ceil(((F.arange(
            self.oshp[1], dtype='float32') + 1) * (input_size[1] / self.oshp[1]))).astype('int32')
        return self._zip(start_points_h, end_points_h), self._zip(start_points_w, end_points_w)

    def _get_windows(self, inp, coords, kernel_size):
        windows = []
        a = 0
        for h_s, h_e in coords[0]:
            for w_s, w_e in coords[1]:
                windows.append(F.mean(inp[:, :, h_s: h_e, w_s: w_e], axis=(2, 3)))
        windows = F.stack(windows, -1)
        return windows

    def forward(self, inputs):
        assert inputs.ndim == 4, "Currently only support 4D input"
        ishp = inputs.shape[-2:]
        kernel_size = self._cal_kernel_size(ishp)
        point_h, point_w = self._get_points(ishp, kernel_size)
        windows = self._get_windows(inputs, (point_h, point_w), kernel_size)
        return windows.reshape(*windows.shape[:2], *self.oshp)