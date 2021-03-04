import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseNet(nn.Module):
    """
    A class to represent a classic pose regressor (PoseNet) with an efficient-net backbone
    PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization,
    Kendall et al., 2015
    """
    def __init__(self, backbone_path):
        """
        Constructor
        :param backbone_path: backbone path to a resnet backbone
        """
        super(PoseNet, self).__init__()

        # Efficient net
        self.backbone = torch.load(backbone_path)
        backbone_dim = 1280
        latent_dim = 1024

        # Regressor layers
        self.fc1 = nn.Linear(backbone_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 3)
        self.fc3 = nn.Linear(latent_dim, 4)

        self.dropout = nn.Dropout(p=0.1)
        self.avg_pooling_2d = nn.AdaptiveAvgPool2d(1)

        # Initialize FC layers
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, data):
        """
        Forward pass
        :param data: (torch.Tensor) dictionary with key-value 'img' -- input image (N X C X H X W)
        :return: (torch.Tensor) dictionary with key-value 'pose' -- 7-dimensional absolute pose for (N X 7)
        """
        x = self.backbone.extract_features(data.get('img'))
        x = self.avg_pooling_2d(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        p_x = self.fc2(x)
        p_q = self.fc3(x)
        return {'pose': torch.cat((p_x, p_q), dim=1)}

