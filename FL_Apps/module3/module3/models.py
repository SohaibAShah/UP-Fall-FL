import torch
import torch.nn as nn
import torch.nn.functional as F

# --- ENCODERS ---
class IMUEncoder(nn.Module):
    def __init__(self, input_channels, feature_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding='same')
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding='same')
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, feature_dim)
    def forward(self, x):
        x = F.relu(self.conv1(x)); x = F.relu(self.conv2(x))
        return self.fc(self.pool(x).squeeze(2))

class ImageEncoder(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, feature_dim)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)); x = F.relu(F.max_pool2d(self.conv2(x), 2))
        return self.fc(self.pool(x).view(x.size(0), -1))

# --- FUSION MODELS ---
class EarlyFusionModel(nn.Module):
    def __init__(self, num_csv_features, num_classes=2):
        super().__init__()
        self.imu_encoder = IMUEncoder(num_csv_features, 64)
        self.img_encoder1 = ImageEncoder(64); self.img_encoder2 = ImageEncoder(64)
        self.classifier = nn.Sequential(nn.Linear(64 + 64 + 64, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes))
    def forward(self, x_csv, x_img1, x_img2):
        f_csv = self.imu_encoder(x_csv); f_img1 = self.img_encoder1(x_img1); f_img2 = self.img_encoder2(x_img2)
        return self.classifier(torch.cat((f_csv, f_img1, f_img2), dim=1))

class LateFusionModel(nn.Module):
    def __init__(self, num_csv_features, num_classes=2):
        super().__init__()
        self.imu_branch = nn.Sequential(IMUEncoder(num_csv_features), nn.Linear(64, num_classes))
        self.img_branch1 = nn.Sequential(ImageEncoder(), nn.Linear(64, num_classes))
        self.img_branch2 = nn.Sequential(ImageEncoder(), nn.Linear(64, num_classes))
        self.fusion_layer = nn.Linear(num_classes * 3, num_classes)
    def forward(self, x_csv, x_img1, x_img2):
        p_csv = self.imu_branch(x_csv); p_img1 = self.img_branch1(x_img1); p_img2 = self.img_branch2(x_img2)
        return self.fusion_layer(torch.cat((p_csv, p_img1, p_img2), dim=1))

class GatedResidualFusionModel(nn.Module):
    def __init__(self, num_csv_features, num_classes=2):
        super().__init__()
        self.imu_encoder = IMUEncoder(num_csv_features, 128)
        self.img_encoder1 = ImageEncoder(64); self.img_encoder2 = ImageEncoder(64)
        self.img_fusion = nn.Linear(64 + 64, 128)
        self.gate = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.fused_classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes))
        self.imu_only_classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes))
    def forward(self, x_csv, x_img1, x_img2, threshold=0.5):
        f_csv = self.imu_encoder(x_csv); gate_prob = torch.sigmoid(self.gate(f_csv))
        if self.training:
            f_img_combined = F.relu(self.img_fusion(torch.cat((self.img_encoder1(x_img1), self.img_encoder2(x_img2)), dim=1)))
            out_fused = self.fused_classifier(f_csv + f_img_combined)
            out_imu_only = self.imu_only_classifier(f_csv)
            return gate_prob * out_fused + (1 - gate_prob) * out_imu_only
        else:
            use_images = (gate_prob > threshold).float()
            out_fused = torch.zeros(f_csv.size(0), 2, device=f_csv.device)
            if use_images.sum() > 0:
                f_img_combined = F.relu(self.img_fusion(torch.cat((self.img_encoder1(x_img1), self.img_encoder2(x_img2)), dim=1)))
                out_fused = self.fused_classifier(f_csv + f_img_combined)
            out_imu_only = self.imu_only_classifier(f_csv)
            return use_images * out_fused + (1 - use_images) * out_imu_only

# --- PERSONALIZATION ---
class Adapter(nn.Module):
    def __init__(self, input_dim=64, bottleneck_dim=8):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(input_dim, bottleneck_dim), nn.ReLU(), nn.Linear(bottleneck_dim, input_dim))
    def forward(self, x): return x + self.block(x)

class PersonalizedAdapterModel(nn.Module):
    def __init__(self, backbone, adapter):
        super().__init__()
        self.backbone = backbone; self.adapter = adapter
        self.classifier = nn.Linear(backbone.feature_dim, 2)
    def forward(self, x_csv, x_img1, x_img2): # Accepts all inputs but only uses CSV for now
        return self.classifier(self.adapter(self.backbone(x_csv)))
