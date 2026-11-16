"""Plot Faster R-CNN and RT-DETR from the epoch table in report.md.

This reproduces reported values; it does not recover or verify training logs.
Run: python notebooks/plot_validation_curves.py
"""
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    repo = Path(__file__).resolve().parents[1]
    text = (repo / 'report.md').read_text(encoding='utf-8')
    table = text.split('| Epoch | Faster R-CNN | RT-DETR |', 1)[1].split('\n\n', 1)[0]
    rows = re.findall(r'^\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|', table, re.MULTILINE)
    epochs = [int(r[0]) for r in rows]
    if epochs != list(range(1, 13)):
        raise ValueError('Expected the complete epoch table, 1 through 12')
    frcnn = [float(r[1]) for r in rows]
    detr = [float(r[2]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.2), layout='constrained')
    ax.plot(epochs, frcnn, marker='o', markersize=4, label='Faster R-CNN (800 / max 1333)')
    ax.plot(epochs, detr, marker='s', markersize=4, label='RT-DETR (640)')
    ax.set(title='Faster R-CNN vs RT-DETR: validation curves', xlabel='Epoch',
           ylabel='Reported validation mAP (IoU 0.50–0.95)', ylim=(0, .25), xticks=epochs)
    ax.grid(alpha=.25)
    ax.legend(loc='lower right', fontsize=8)
    target = repo / 'images' / 'validation_frcnn_rtdetr.png'
    fig.savefig(target, dpi=160, metadata={'Description': 'Source: the 12-epoch table in report.md; training logs were not used.'})
    plt.close(fig)
    print(target)
    for name, values in [('Faster R-CNN', frcnn), ('RT-DETR', detr)]:
        print(f'{name}: best {max(values):.4f} at epoch {epochs[values.index(max(values))]}; final {values[-1]:.4f}')


if __name__ == '__main__':
    main()
