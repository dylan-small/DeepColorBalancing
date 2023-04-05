import torch.nn as nn
import torch
import numpy as np
import kornia

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

def rgbToHue(rgb):
    return rgbToHsv(rgb)[:, 0]

# def rgbToLab(rgb):
#     # out = kornia.color.rgb_to_lab(rgb[:,:,None,None]).squeeze()
#     transform = kornia.color.RgbToLab()
#     out = transform(rgb[:,:,None,None]).squeeze()
#     return out

def rgbToLab(rgb):
    R = rgb[:,0]
    G = rgb[:,1]
    B = rgb[:,2]
    L = (13933 * R + 46871 * G + 4732 * B) / 2**16
    A = 377 * (14503 * R - 22218 * G + 7714 * B) / 2**24 + 128
    b = (12773 * R + 39695 * G - 52468 * B) / 2**24 + 128
    out = torch.stack([L, A, b], dim=1)
    return out


spaces = {
    'RGB': None,
    'HSV': rgbToHsv,
    'Hue': rgbToHue,
    'LAB': rgbToLab,
}
class Criterion(nn.Module):
    def __init__(self, loss='RMSE', space='RGB'):
        super(Criterion, self).__init__()
        assert loss in losses
        assert space in spaces
        self.space = space
        self.loss = losses[loss]

    def transformToSpace(self, rgb):
        if len(rgb.shape) == 1:
            rgb = rgb[None, :]
        out = spaces[self.space](rgb)
        if len(out.shape) == 1:
            out = out[None, :]
        return out
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
    rgbToLab(colors)