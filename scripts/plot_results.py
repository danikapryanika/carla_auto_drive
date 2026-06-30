"""
Генерация графиков для дипломной работы.

Использование:
    # После сбора данных:
    python scripts/plot_results.py --mode dataset

    # После обучения (читает TensorBoard логи):
    python scripts/plot_results.py --mode training

    # Всё сразу:
    python scripts/plot_results.py --mode all

Все графики сохраняются в папку plots/
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # без GUI — для сохранения в файл
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PLOTS_DIR   = "plots"
DATA_CSV    = os.path.join("data", "train", "labels.csv")
TENSORBOARD = os.path.join("logs", "tensorboard")

CMD_NAMES = {0: "FOLLOW", 1: "LEFT", 2: "RIGHT", 3: "STRAIGHT"}
CMD_COLORS = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868", 3: "#C44E52"}

os.makedirs(PLOTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f" {path}")


# ─────────────────────────────────────────────────────────────────────────────
# DATASET plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_dataset(csv_path=DATA_CSV):
    print("Генерация графиков датасета …")
    df = pd.read_csv(csv_path)
    n  = len(df)
    print(f"   Всего кадров: {n}")

    # ── 1. Command distribution (pie + bar) ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Распределение навигационных команд в датасете", fontsize=14, fontweight='bold')

    cmd_counts = df['command'].value_counts().sort_index()
    labels  = [CMD_NAMES.get(i, str(i)) for i in cmd_counts.index]
    colors  = [CMD_COLORS.get(i, '#999') for i in cmd_counts.index]

    axes[0].pie(cmd_counts.values, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=90,
                wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    axes[0].set_title("Доля команд (pie chart)")

    axes[1].bar(labels, cmd_counts.values, color=colors, edgecolor='black', linewidth=0.7)
    axes[1].set_xlabel("Команда")
    axes[1].set_ylabel("Количество кадров")
    axes[1].set_title("Количество кадров по командам")
    for i, v in enumerate(cmd_counts.values):
        axes[1].text(i, v + 20, f'{v}\n({100*v/n:.1f}%)', ha='center', fontsize=9)

    plt.tight_layout()
    save(fig, "01_command_distribution.png")

    # ── 2. Steering angle distribution ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Распределение угла руля (steer)", fontsize=14, fontweight='bold')

    axes[0].hist(df['steer'], bins=80, color='steelblue', edgecolor='black',
                 linewidth=0.4, alpha=0.85)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=1.2, label='центр')
    axes[0].set_xlabel("Угол руля [-1, 1]")
    axes[0].set_ylabel("Количество кадров")
    axes[0].set_title("Полное распределение")
    axes[0].legend()

    # Per-command overlay
    for cmd_id, name in CMD_NAMES.items():
        sub = df[df['command'] == cmd_id]['steer']
        if len(sub) > 0:
            axes[1].hist(sub, bins=60, alpha=0.6, label=name,
                         color=CMD_COLORS[cmd_id], edgecolor='none')
    axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[1].set_xlabel("Угол руля [-1, 1]")
    axes[1].set_ylabel("Количество кадров")
    axes[1].set_title("По командам")
    axes[1].legend()

    plt.tight_layout()
    save(fig, "02_steer_distribution.png")

    # ── 3. Throttle & speed distributions ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Распределение газа и скорости", fontsize=14, fontweight='bold')

    axes[0].hist(df['throttle'], bins=60, color='#55A868', edgecolor='black', linewidth=0.4)
    axes[0].set_xlabel("Throttle [0, 1]")
    axes[0].set_ylabel("Количество кадров")
    axes[0].set_title("Газ (throttle)")

    axes[1].hist(df['speed_kmh'], bins=60, color='#C44E52', edgecolor='black', linewidth=0.4)
    axes[1].set_xlabel("Скорость (км/ч)")
    axes[1].set_ylabel("Количество кадров")
    axes[1].set_title("Скорость автомобиля")

    plt.tight_layout()
    save(fig, "03_throttle_speed_distribution.png")

    # ── 4. Steer timeline (первые 2000 кадров) ────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Траектория управления (первые 2 000 кадров)", fontsize=14, fontweight='bold')

    sample = df.head(2000)
    axes[0].plot(sample.index, sample['steer'],    color='steelblue', linewidth=0.8)
    axes[0].axhline(0, color='red', linestyle='--', linewidth=0.8)
    axes[0].set_ylabel("Руль (steer)")
    axes[0].set_ylim(-1.1, 1.1)

    axes[1].plot(sample.index, sample['throttle'], color='green',     linewidth=0.8)
    axes[1].set_ylabel("Газ (throttle)")
    axes[1].set_ylim(-0.05, 1.05)

    axes[2].plot(sample.index, sample['speed_kmh'], color='orange',   linewidth=0.8)
    axes[2].set_ylabel("Скорость (км/ч)")
    axes[2].set_xlabel("Кадр")

    plt.tight_layout()
    save(fig, "04_control_timeline.png")

    # ── 5. Correlation: steer vs speed ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(df['speed_kmh'], df['steer'].abs(),
                    c=df['command'], cmap='tab10', alpha=0.2, s=2,
                    vmin=0, vmax=3)
    from matplotlib.lines import Line2D
    legend_els = [Line2D([0], [0], marker='o', color='w',
                         markerfacecolor=plt.cm.tab10(i / 10), markersize=8,
                         label=CMD_NAMES[i]) for i in range(4)]
    ax.legend(handles=legend_els, title="Команда")
    ax.set_xlabel("Скорость (км/ч)")
    ax.set_ylabel("|Руль| (абс. значение)")
    ax.set_title("Зависимость угла руля от скорости (по командам)")
    plt.tight_layout()
    save(fig, "05_steer_vs_speed.png")

    print(f"Графики датасета сохранены в {PLOTS_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING plots (from TensorBoard event files)
# ─────────────────────────────────────────────────────────────────────────────

def _read_tb(logdir):
    """Read scalar summaries from TensorBoard event files."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("'tensorboard' не установлен для чтения логов напрямую.")
        print("   Запусти: tensorboard --logdir logs/tensorboard")
        return None

    ea = EventAccumulator(logdir)
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    data = {}
    for tag in tags:
        events   = ea.Scalars(tag)
        data[tag] = {
            'step':  np.array([e.step  for e in events]),
            'value': np.array([e.value for e in events]),
        }
    return data


def plot_training(logdir=TENSORBOARD):
    print("Генерация графиков обучения …")
    data = _read_tb(logdir)
    if data is None:
        return

    # ── 6. Train / Val Loss ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Кривые обучения CILDriveNet", fontsize=14, fontweight='bold')

    def plot_pair(ax, train_tag, val_tag, title, ylabel):
        if train_tag in data:
            ax.plot(data[train_tag]['step'], data[train_tag]['value'],
                    label='Train', color='steelblue', linewidth=1.5)
        if val_tag in data:
            ax.plot(data[val_tag]['step'], data[val_tag]['value'],
                    label='Val', color='#C44E52', linewidth=1.5)
        ax.set_title(title); ax.set_xlabel("Эпоха"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True, alpha=0.3)

    plot_pair(axes[0], 'Loss/train', 'Loss/val',     'Суммарный Loss',   'Loss')
    plot_pair(axes[1], 'Steer/train', 'Steer/val',   'Steer Loss',       'Loss')
    plot_pair(axes[2], 'Throttle/train', 'Throttle/val', 'Throttle Loss', 'Loss')

    plt.tight_layout()
    save(fig, "06_training_curves.png")

    # ── 7. Per-command validation loss ────────────────────────────────────────
    cmd_tags = ['Val/cmd_0', 'Val/cmd_1', 'Val/cmd_2', 'Val/cmd_3']
    available = [t for t in cmd_tags if t in data]
    if available:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle("Валидационный Loss по командам (branch losses)",
                     fontsize=14, fontweight='bold')
        for tag in available:
            cmd_id = int(tag.split('_')[-1])
            ax.plot(data[tag]['step'], data[tag]['value'],
                    label=CMD_NAMES[cmd_id], color=CMD_COLORS[cmd_id], linewidth=1.8)
        ax.set_xlabel("Эпоха"); ax.set_ylabel("Loss")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save(fig, "07_per_command_loss.png")

    print(f"Графики обучения сохранены в {PLOTS_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
# Architecture diagram (схема модели для диплома)
# ─────────────────────────────────────────────────────────────────────────────

def plot_architecture():
    print("Генерация схемы архитектуры …")

    fig = plt.figure(figsize=(16, 8))
    ax  = fig.add_subplot(111)
    ax.set_xlim(0, 16); ax.set_ylim(0, 8)
    ax.axis('off')
    fig.suptitle("Архитектура CILDriveNet (Conditional Imitation Learning)",
                 fontsize=14, fontweight='bold')

    def box(ax, x, y, w, h, label, sub='', color='#AED6F1', fontsize=9):
        rect = plt.Rectangle((x, y), w, h, linewidth=1.5,
                              edgecolor='#2C3E50', facecolor=color, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.15 if sub else 0), label,
                ha='center', va='center', fontsize=fontsize, fontweight='bold', zorder=4)
        if sub:
            ax.text(x + w/2, y + h/2 - 0.25, sub,
                    ha='center', va='center', fontsize=7, color='#555', zorder=4)

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.5), zorder=5)

    # Inputs
    box(ax, 0.2, 5.5, 2.0, 1.0, 'Camera',       '3×224×224', '#AED6F1')
    box(ax, 0.2, 3.5, 2.0, 1.0, 'LIDAR BEV',    '5×200×200', '#A9DFBF')
    box(ax, 0.2, 1.5, 2.0, 1.0, 'Speed',        '1 (норм.)', '#F9E79F')

    # Encoders
    box(ax, 3.0, 5.5, 2.2, 1.0, 'ResNet-18',    'pretrained → 512d', '#AED6F1')
    box(ax, 3.0, 3.5, 2.2, 1.0, 'LidarCNN',     '3×Conv → 128d',    '#A9DFBF')
    box(ax, 3.0, 1.5, 2.2, 1.0, 'SpeedMLP',     'FC(1→64)',          '#F9E79F')

    # Fusion
    box(ax, 6.2, 3.5, 2.0, 1.0, 'Fusion',       'FC(704→256)', '#E8DAEF')

    # Command
    box(ax, 6.0, 1.2, 2.2, 0.9, 'Command',      '0-3 (карта)', '#FDEBD0')

    # Branches
    bcolors = ['#AED6F1', '#A9DFBF', '#F9E79F', '#FAD7A0']
    bnames  = ['Branch 0\nFOLLOW', 'Branch 1\nLEFT',
               'Branch 2\nRIGHT',  'Branch 3\nSTRAIGHT']
    for i, (name, col) in enumerate(zip(bnames, bcolors)):
        yb = 6.5 - i * 1.5
        box(ax, 9.5, yb, 2.2, 1.0, name, 'FC(256→128→2)', col)
        arrow(ax, 8.2, 4.0, 9.5, yb + 0.5)

    # Output
    box(ax, 12.5, 3.5, 2.5, 1.0, 'Output',
        '[steer, throttle]', '#FADBD8', fontsize=10)
    for i in range(4):
        yb = 6.5 - i * 1.5 + 0.5
        arrow(ax, 11.7, yb, 12.5, 4.0)

    # Arrows encoders → fusion
    arrow(ax, 2.2, 6.0, 3.0, 6.0)
    arrow(ax, 2.2, 4.0, 3.0, 4.0)
    arrow(ax, 2.2, 2.0, 3.0, 2.0)
    arrow(ax, 5.2, 6.0, 6.2, 4.3)
    arrow(ax, 5.2, 4.0, 6.2, 4.0)
    arrow(ax, 5.2, 2.0, 6.2, 3.7)

    # Command → selector hint
    ax.text(8.2, 1.65, '← активная\n   ветвь', fontsize=8, color='#8B0000')

    plt.tight_layout()
    save(fig, "00_architecture.png")
    print(f"Схема архитектуры сохранена в {PLOTS_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    if args.mode in ('dataset', 'all'):
        if os.path.exists(DATA_CSV):
            plot_dataset(DATA_CSV)
        else:
            print(f"Не найден {DATA_CSV} — сначала соберите данные.")

    if args.mode in ('training', 'all'):
        if os.path.isdir(TENSORBOARD):
            plot_training(TENSORBOARD)
        else:
            print(f"Не найден {TENSORBOARD} — сначала запустите обучение.")

    if args.mode in ('arch', 'all'):
        plot_architecture()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Графики для дипломной работы")
    parser.add_argument('--mode', choices=['dataset', 'training', 'arch', 'all'],
                        default='all', help="Что генерировать")
    main(parser.parse_args())
