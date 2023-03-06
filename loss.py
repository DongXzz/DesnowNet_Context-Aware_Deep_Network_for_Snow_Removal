import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models

def weight_decay_l2(loss, model, lambda_w):
    wdecay = 0
    for w in model.parameters():
        if w.requires_grad:
            wdecay = torch.add(torch.sum(w ** 2), wdecay)

    loss = torch.add(loss, lambda_w * wdecay)
    return loss


def lw_pyramid_loss(m, hat_m, tau=6):
    """
     lightweight pyramid loss function
    :param m: one image
    :param hat_m: another image of the same size
    :param tau:the level of loss pyramid, default 6
    :return: loss
    """
    # try use perception loss
    batch_size = m.shape[0]
    loss = 0
    for i in range(tau + 1):
        block = nn.MaxPool2d(2**i, stride=2**i)
        p1 = block(m)
        p2 = block(hat_m)
        loss += torch.sum((p1-p2)**2)
    return loss/batch_size

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

def style_loss(A_feats, B_feats):
    assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
    loss_value = 0.0
    for i in range(len(A_feats)):
        A_feat = A_feats[i]
        B_feat = B_feats[i]
        _, c, w, h = A_feat.size()
        A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
        B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
        A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
        B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
        loss_value += torch.mean(torch.abs(A_style - B_style)/(c * w * h))
    return loss_value
    
def preceptual_loss(A_feats, B_feats):
    assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
    loss_value = 0.0
    for i in range(len(A_feats)):
        A_feat = A_feats[i]
        B_feat = B_feats[i]
        loss_value += torch.mean(torch.abs(A_feat - B_feat))
    return loss_value

if __name__ == '__main__':
    device = 'cuda'
    img1 = torch.zeros([5, 1, 64, 64], device=device).requires_grad_()
    img2 = torch.ones_like(img1, device=device).requires_grad_() * 0.1
    loss = lw_pyramid_loss(img1, img2)
    module1 = nn.Conv2d(3,128,3)
    loss = weight_decay_l2(loss, module1, 0.2)
    print("finished")