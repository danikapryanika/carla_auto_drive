"""
CarlaDataset — загружает последовательности кадров для GRU-обучения.

Каждый сэмпл содержит:
  imgs_seq  (seq_len, 3, 224, 224)  — T последовательных кадров одной камеры
  lidar_bev (5, 200, 200)           — LiDAR текущего (последнего) кадра
  speed_t   (1,)                    — скорость текущего кадра
  command   int                      — навигационная команда текущего кадра
  targets   (2,)                    — [steer, throttle] текущего кадра

Последовательность строится из строк CSV с тем же 'cam' (C/L/R).
Если предыдущих кадров не хватает — повторяем самый ранний доступный.

seq_len=1 → поведение как у старого датасета (без GRU).
"""

import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from utils import IMG_MEAN, IMG_STD, MAX_SPEED_KMH

SEQ_LEN = 4   # кадров в последовательности (0.2 сек при 20 FPS)


class CarlaDataset(Dataset):
    def __init__(self, csv_path: str, data_dir: str,
                 train: bool = True, seq_len: int = SEQ_LEN):
        self.df        = pd.read_csv(csv_path)
        self.img_dir   = os.path.join(data_dir, "images")
        self.lidar_dir = os.path.join(data_dir, "lidar")
        self.train     = train
        self.seq_len   = seq_len

        # Проверяем наличие колонки cam (обратная совместимость со старым датасетом)
        if 'cam' not in self.df.columns:
            self.df['cam'] = 'C'

        # Индексы строк для каждой камеры: {'C': [0,3,6,...], 'L': [1,4,...], 'R': [2,5,...]}
        self._cam_idx = {}
        for cam in self.df['cam'].unique():
            self._cam_idx[cam] = list(self.df.index[self.df['cam'] == cam])

        # Позиция каждой строки внутри своей камеры: row_i → pos внутри cam-группы
        self._row_pos = {}
        for cam, indices in self._cam_idx.items():
            for pos, idx in enumerate(indices):
                self._row_pos[idx] = pos

        # ── Sampling weights ─────────────────────────────────────────────────
        w = torch.ones(len(self.df), dtype=torch.float32)
        w[self.df['steer'].abs() > 0.06] *= 8.0
        w[self.df['command'].isin([1, 2, 3])] *= 6.0
        self.weights = w

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, row) -> np.ndarray:
        img_bgr = cv2.imread(os.path.join(self.img_dir, row['image_path']))
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    def _augment(self, img: np.ndarray, lidar_bev: np.ndarray,
                 steer: float, command: int):
        """Аугментации применяются одинаково ко всем кадрам последовательности."""
        # Горизонтальный флип (одно решение на всю последовательность)
        do_flip = self.train and np.random.random() > 0.5
        # Цветовое дрожание (отдельное для каждого кадра)
        alpha = np.random.uniform(0.80, 1.20) if self.train else 1.0
        beta  = np.random.uniform(-20.0, 20.0) if self.train else 0.0
        do_hsv = self.train and np.random.random() > 0.6
        do_lidar_drop = self.train and np.random.random() > 0.8
        return do_flip, alpha, beta, do_hsv, do_lidar_drop

    def _process_image(self, img: np.ndarray, do_flip: bool,
                       alpha: float, beta: float, do_hsv: bool) -> torch.Tensor:
        if do_flip:
            img = img[:, ::-1, :].copy()
        img = np.clip(img * alpha + beta, 0.0, 255.0)
        if do_hsv:
            hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 2] *= np.random.uniform(0.70, 1.30)
            hsv[:, :, 2]  = np.clip(hsv[:, :, 2], 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        img /= 255.0
        img  = (img - IMG_MEAN) / IMG_STD
        return torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)  # (3,H,W)

    def __getitem__(self, idx: int):
        row     = self.df.iloc[idx]
        cam     = row['cam']
        cam_idx = self._cam_idx[cam]      # все индексы этой камеры
        pos     = self._row_pos[idx]       # позиция текущей строки в камере

        # ── Построить последовательность индексов ────────────────────────────
        # [pos-(T-1), ..., pos-1, pos], клэмпим на 0
        seq_positions = [max(0, pos - (self.seq_len - 1 - k))
                         for k in range(self.seq_len)]
        seq_df_indices = [cam_idx[p] for p in seq_positions]

        # ── Определить аугментации (одни для всей последовательности) ────────
        cur_row = self.df.iloc[idx]
        steer   = float(cur_row['steer'])
        command = int(cur_row['command'])
        do_flip, alpha, beta, do_hsv, do_lidar_drop = self._augment(
            None, None, steer, command)

        # ── Загрузить и обработать изображения последовательности ────────────
        imgs = []
        for si in seq_df_indices:
            img_raw = self._load_image(self.df.iloc[si])
            img_t   = self._process_image(img_raw, do_flip, alpha, beta, do_hsv)
            imgs.append(img_t)
        imgs_seq = torch.stack(imgs, dim=0)  # (seq_len, 3, H, W)

        # ── LiDAR текущего кадра ─────────────────────────────────────────────
        lidar_bev = np.load(
            os.path.join(self.lidar_dir, cur_row['lidar_path'])
        ).astype(np.float32)
        if do_flip:
            lidar_bev = lidar_bev[:, :, ::-1].copy()
        if do_lidar_drop:
            lidar_bev[np.random.random(lidar_bev.shape) < 0.05] = 0.0
        lidar_t = torch.from_numpy(lidar_bev)  # (5, 200, 200)

        # ── Метки текущего кадра ─────────────────────────────────────────────
        speed   = float(cur_row['speed_kmh']) / MAX_SPEED_KMH
        if do_flip:
            steer = -steer
            if command == 1: command = 2
            elif command == 2: command = 1

        speed_t  = torch.tensor([np.clip(speed, 0.0, 1.0)], dtype=torch.float32)
        targets  = torch.tensor([steer, float(cur_row['throttle'])],
                                dtype=torch.float32)

        return imgs_seq, lidar_t, speed_t, command, targets


def get_dataloaders(data_dir: str, batch_size: int = 16,
                    val_split: float = 0.15, num_workers: int = 2,
                    seq_len: int = SEQ_LEN):
    csv_path = os.path.join(data_dir, "labels.csv")

    trn_full = CarlaDataset(csv_path, data_dir, train=True,  seq_len=seq_len)
    val_full = CarlaDataset(csv_path, data_dir, train=False, seq_len=seq_len)

    n        = len(trn_full)
    trn_size = int(n * (1 - val_split))

    trn_idx = list(range(trn_size))
    val_idx = list(range(trn_size, n))

    trn_ds = torch.utils.data.Subset(trn_full, trn_idx)
    val_ds = torch.utils.data.Subset(val_full, val_idx)

    trn_weights = trn_full.weights[torch.tensor(trn_idx)]
    sampler     = WeightedRandomSampler(trn_weights, len(trn_ds), replacement=True)

    trn_loader = DataLoader(trn_ds, batch_size=batch_size, sampler=sampler,
                            num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    df = trn_full.df
    print(f"Dataset: {n} строк  ({trn_size} train / {n-trn_size} val) | seq_len={seq_len}")
    for cmd_id, name in {0:"FOLLOW",1:"LEFT",2:"RIGHT",3:"STRAIGHT"}.items():
        cnt = (df['command'] == cmd_id).sum()
        print(f"   {name:8s}: {cnt:5d} ({100*cnt/n:.1f}%)")

    return trn_loader, val_loader
