import torch.nn as nn
import torch
import numpy as np

losses = {
    'RMSE': nn.MSELoss()
}

def rgbToHsv(rgb):
    rgb = rgb * 255
    M = torch.max(rgb, dim=1)
    m = torch.min(rgb, dim=1)
    V = M.values / 255
    S = 1 - m.values/M.values
    S[S<0] = 0
    R = rgb[:, 0]
    G = rgb[:, 1]
    B = rgb[:, 2]
    H1 = torch.arccos((R - G / 2 - B / 2) / torch.sqrt(R**2 + G**2 + B**2 - R*G - R*B - G*B)) / np.pi * 180 / 360
    H1 = torch.nan_to_num(H1, 0)
    H2 = 1 - H1
    out = torch.zeros(rgb.shape)
    out[:, 0] = H2
    out[G >= B, 0] = H1[G >= B]
    out[:, 1] = S
    out[:, 2] = V
    return out

spaces = {
    'RGB': None,
    'HSV': rgbToHsv,
    'LAB': 1,
    'XYZ': 1
}
class Criterion(nn.Module):
    def __init__(self, loss='RMSE', space='RGB'):
        super(Criterion, self).__init__()
        assert loss in losses
        assert space in spaces
        self.space = space
        self.loss = losses[loss]

    def transformToSpace(self, rgb):
        return spaces[self.space](rgb)
    def forward(self, pred, target):
        predInSpace = pred
        targetInSpace = target
        if self.space != 'RGB':
            predInSpace = self.transformToSpace(pred)
            targetInSpace = self.transformToSpace(target)
        # both pred, target in rgb
        return self.loss(predInSpace, targetInSpace)


if __name__ == '__main__':
    criterion = Criterion()
    colors = torch.Tensor([[0.5,0.5,0.5],[0.1,0.7,0.3]])
    rgbToHsv(colors)