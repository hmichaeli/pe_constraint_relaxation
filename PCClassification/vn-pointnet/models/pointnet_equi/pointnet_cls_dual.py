import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.pointnet_equi.layers import *
from models.pointnet_equi.pointnet import PointNetEncoder_Dual

class get_model(nn.Module):
    def __init__(self, args, k=40, normal_channel=True):
        super(get_model, self).__init__()
        self.args = args
        # Prefer args.strict_equiv if provided
        strict_equiv = getattr(args, 'strict_equiv', False)
        width = getattr(args, 'width', 1024)
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder_Dual(args, global_feat=True, feature_transform=True, channel=channel, strict_equiv=strict_equiv, width=width)
        self.fc1 = nn.Linear(width//3*6, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x,equiv,mix):
        x, trans, trans_feat,n_ld,n = self.feat(x,equiv,mix)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat,(n_ld,n)

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        #self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        #mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss #+ mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
