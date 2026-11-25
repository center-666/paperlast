"""Plot early alarm probabilities and the paired interaction from results.json."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    here = Path(__file__).resolve().parent
    result = json.loads((here / 'results.json').read_text(encoding='utf-8'))
    horizon = np.arange(1, result['observation_window'] + 1)
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 5.8), sharex=True, layout='constrained',
                             height_ratios=[2, 1])
    colors = {'iid': '#1F5B8C', 'negative': '#A64428'}
    for column, (model, data) in enumerate(result['models'].items()):
        early = data['summary']['early_alarm_cdf_h1_to_h8']
        top, bottom = axes[:, column]
        for arm, watermark, style, name in (
            ('A', 'iid', '--', 'KL, iid'),
            ('B', 'negative', '--', 'KL, negative coupling'),
            ('C', 'iid', '-', 'Matched, iid'),
            ('D', 'negative', '-', 'Matched, negative coupling'),
        ):
            top.plot(horizon, 100 * np.asarray(early['arm_mean'][arm]), style, marker='o',
                     markersize=3.5, linewidth=1.7, color=colors[watermark], label=name)
        top.set_title(f'{model.capitalize()} dynamics', loc='left', fontweight='bold')
        top.set_ylim(0, 103)
        top.set_yticks(np.arange(0, 101, 25))
        top.legend(loc='lower right', fontsize=8, frameon=False)
        contrast = early['contrasts']['interaction']
        lo, hi = 100 * np.asarray(contrast['ci95'])
        bottom.fill_between(horizon, lo, hi, color='#DBD5E7', linewidth=0)
        bottom.plot(horizon, 100 * np.asarray(contrast['mean']), '-o',
                    markersize=3.5, linewidth=1.7, color='#554375')
        bottom.axhline(0, color='#555555', linewidth=0.75)
        bottom.set_xlabel('Packets since attack onset')
        bottom.set_xticks(horizon)
        for ax in (top, bottom):
            ax.axvline(result['primary_h'], color='#9C9C9C', linestyle=':', linewidth=0.9)
            ax.grid(axis='y', color='#E7E7E7', linewidth=0.7)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_axisbelow(True)
    axes[0, 0].set_ylabel('Cumulative alarm probability (%)')
    axes[1, 0].set_ylabel('Paired interaction (pp)')
    fig.savefig(here / 'early_alarms.pdf')


if __name__ == '__main__':
    main()
