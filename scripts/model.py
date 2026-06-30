"""
CILDriveNet — мультимодальная CIL-модель с GRU для временной памяти.

Архитектура:
  Camera  (ResNet-18, ImageNet pretrained)  -> 512-d
  LiDAR   (CNN, 5x200x200 BEV)              -> 128-d
  Speed   (MLP, 1->64)                      -> 64-d
  Fusion  FC(704->256)
  GRU     (256->256)                        <- запоминает историю T кадров
  Branches x4: FC(256->128->2) [steer=tanh, throttle=sigmoid]

При обучении: imgs_seq (B, T, 3, H, W) — T последовательных кадров.
При инференсе: imgs_seq (B, 1, 3, H, W) или (B, 3, H, W), h передаётся между тиками.
"""

import torch
import torch.nn as nn

try:
    from torchvision.models import resnet18, ResNet18_Weights
    def _resnet18():
        return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
except (ImportError, AttributeError):
    from torchvision import models
    def _resnet18():
        return models.resnet18(pretrained=True)


# ── Sub-modules ───────────────────────────────────────────────────────────────

class LidarEncoder(nn.Module):
    """Tiny CNN: (5, 200, 200) BEV -> 128-d vector."""
    def __init__(self, in_ch: int = 5, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,     32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.Conv2d(32,        64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Conv2d(64, out_dim,  3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
    def forward(self, x): return self.net(x)


class SpeedEncoder(nn.Module):
    """MLP: scalar speed [0,1] -> 64-d vector."""
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, out_dim), nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


# ── Main model ────────────────────────────────────────────────────────────────

class CILDriveNet(nn.Module):
    """
    Conditional Imitation Learning + GRU temporal memory.

    Inputs:
        imgs_seq  (B, T, 3, 224, 224)  -- T sequential camera frames
        lidar_bev (B, 5, 200, 200)     -- LiDAR BEV of current frame
        speed     (B, 1)               -- normalised speed [0,1]
        h         (1, B, 256) or None  -- GRU hidden state (None = zeros)

    Outputs:
        actions    (B, 4, 2)   -- steer (tanh) + throttle (sigmoid) per command
        speed_pred (B, 1)      -- predicted speed (auxiliary loss, sigmoid)
        h_new      (1, B, 256) -- updated GRU hidden state (keep between ticks)

    Command mapping:  0=FOLLOW  1=LEFT  2=RIGHT  3=STRAIGHT
    """

    NUM_COMMANDS = 4
    CAM_DIM      = 512
    LIDAR_DIM    = 128
    SPEED_DIM    = 64
    FUSE_DIM     = CAM_DIM + LIDAR_DIM + SPEED_DIM   # 704
    GRU_DIM      = 256

    def __init__(self):
        super().__init__()

        resnet = _resnet18()
        self.camera_encoder = nn.Sequential(*list(resnet.children())[:-1])

        self.lidar_encoder = LidarEncoder(in_ch=5, out_dim=self.LIDAR_DIM)
        self.speed_encoder = SpeedEncoder(out_dim=self.SPEED_DIM)

        self.fusion = nn.Sequential(
            nn.Linear(self.FUSE_DIM, 512), nn.ReLU(inplace=True),
            nn.Linear(512, self.GRU_DIM),  nn.ReLU(inplace=True),
        )

        # GRU: обрабатывает последовательность fusion-векторов
        self.gru = nn.GRU(
            input_size=self.GRU_DIM,
            hidden_size=self.GRU_DIM,
            num_layers=1,
            batch_first=False,   # (T, B, GRU_DIM)
        )
        self.gru_dropout = nn.Dropout(p=0.3)

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.GRU_DIM, 128), nn.ReLU(inplace=True),
                nn.Linear(128, 2),
            )
            for _ in range(self.NUM_COMMANDS)
        ])

        # Вспомогательная голова предсказания скорости из визуальных признаков
        self.speed_head = nn.Sequential(
            nn.Linear(self.CAM_DIM + self.LIDAR_DIM, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, imgs_seq, lidar_bev, speed, h=None):
        # Поддержка одиночного кадра (B, 3, H, W) -> (B, 1, 3, H, W)
        if imgs_seq.dim() == 4:
            imgs_seq = imgs_seq.unsqueeze(1)

        B, T, C, H, W = imgs_seq.shape

        # ── Закодировать все T кадров камерой ────────────────────────────────
        imgs_flat = imgs_seq.reshape(B * T, C, H, W)
        cam_f_all = self.camera_encoder(imgs_flat).flatten(1)   # (B*T, 512)

        # ── LiDAR и скорость — текущий кадр, растягиваем на T ────────────────
        lidar_f = self.lidar_encoder(lidar_bev)   # (B, 128)
        speed_f = self.speed_encoder(speed)        # (B, 64)

        lidar_f_all = lidar_f.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
        speed_f_all = speed_f.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)

        # ── Aux: предсказать скорость из текущего кадра ───────────────────────
        cam_f_cur  = cam_f_all.view(B, T, -1)[:, -1, :]   # (B, 512)
        speed_pred = self.speed_head(torch.cat([cam_f_cur, lidar_f], dim=1))

        # ── Fusion: конкатенация → FC ─────────────────────────────────────────
        fused = torch.cat([cam_f_all, lidar_f_all, speed_f_all], dim=1)  # (B*T, 704)
        fused = self.fusion(fused)                                         # (B*T, 256)

        # ── GRU ───────────────────────────────────────────────────────────────
        fused = fused.view(B, T, self.GRU_DIM).permute(1, 0, 2)  # (T, B, 256)
        gru_out, h_new = self.gru(fused, h)                        # (T,B,256), (1,B,256)
        x = self.gru_dropout(gru_out[-1])                          # (B, 256)

        # ── Командные ветки ───────────────────────────────────────────────────
        branch_outs = []
        for branch in self.branches:
            raw      = branch(x)
            steer    = torch.tanh(raw[:, 0:1])
            throttle = torch.sigmoid(raw[:, 1:2])
            branch_outs.append(torch.cat([steer, throttle], dim=1))

        actions = torch.stack(branch_outs, dim=1)   # (B, 4, 2)
        return actions, speed_pred, h_new
