import torch
import torch.nn as nn
import torchvision.models as models

class ImageCritic(nn.Module):
    def __init__(self, camera_names, state_dim=21, hidden_dim=256):
        super().__init__()
        self.camera_names = camera_names
        
        # 1. 为每个相机创建一个轻量级的视觉编码器 (这里用 ResNet18 的特征层)
        self.visual_encoders = nn.ModuleDict()
        for cam in camera_names:
            resnet = models.resnet18(weights=None) # 微调时通常从头训 Critic 或冻结预训练特征
            # 去掉最后的全连接层，输出维度为 512
            self.visual_encoders[cam] = nn.Sequential(*list(resnet.children())[:-1])
            
        # 2. 状态编码器
        self.state_encoder = nn.Linear(state_dim, hidden_dim)
        
        # 3. 融合后的全连接层 (输出一个标量 Value)
        # 输入维度: len(cameras) * 512 (图像特征) + hidden_dim (状态特征)
        total_feature_dim = len(camera_names) * 512 + hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(total_feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # 🌟 Critic 的核心：输出唯一的价值评估
        )

    def forward(self, batch):
        """
        接收与 Actor 相同的 batch 字典输入
        """
        features = []
        
        # 提取图像特征
        for cam in self.camera_names:
            img_tensor = batch[f'observation.images.{cam}'] # [BS, C, H, W]
            # ResNet 输出是 [BS, 512, 1, 1]，压平为 [BS, 512]
            feat = self.visual_encoders[cam](img_tensor).squeeze(-1).squeeze(-1)
            features.append(feat)
            
        # 提取状态特征
        state_tensor = batch['observation.state'] # [BS, 21]
        state_feat = self.state_encoder(state_tensor)
        features.append(state_feat)
        
        # 拼接所有特征
        concat_features = torch.cat(features, dim=-1)
        
        # 计算价值 V(s)
        value = self.mlp(concat_features) # [BS, 1]
        return value.squeeze(-1) # 返回 [BS]