"""Generate cooperation trajectory figure for Atom Hybrid paper.

Usage: python3 plot_trajectory.py
Output: figures/cooperation_trajectory.png, figures/cooperation_trajectory.pdf
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# Data extracted from /tmp/run{7,8,9}_output.txt (verified via spotcheck)
runs = {
    'Run 7 (300t)': {
        'ticks': [60, 120, 180, 240, 300],
        'A': [100, 98, 98, 100, 100],
        'B': [100, 98, 100, 96, 98],
        'C': [83, 100, 100, 100, 100],
    },
    'Run 8 (200t)': {
        'ticks': [40, 80, 120, 160, 200],
        'A': [100, 100, 100, 100, 100],
        'B': [94, 97, 100, 97, 100],
        'C': [84, 88, 100, 100, 100],
    },
    'Run 9 (200t)': {
        'ticks': [40, 80, 120, 160, 200],
        'A': [100, 100, 100, 99, 99],
        'B': [100, 94, 94, 91, 100],
        'C': [62, 87, 100, 96, 100],
    },
}

condition_styles = {
    'A': {'color': '#888888', 'linestyle': '-',  'label_prefix': 'A: Single LLM'},
    'B': {'color': '#3b82f6', 'linestyle': '-',  'label_prefix': 'B: LLM Society'},
    'C': {'color': '#dc2626', 'linestyle': '-',  'label_prefix': 'C: Atom Hybrid'},
}

run_markers = {'Run 7 (300t)': 'o', 'Run 8 (200t)': 's', 'Run 9 (200t)': '^'}

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
plt.subplots_adjust(wspace=0.08)

for ax, (run_name, data) in zip(axes, runs.items()):
    ticks = data['ticks']
    for cond in ['A', 'B', 'C']:
        s = condition_styles[cond]
        ax.plot(ticks, data[cond],
                color=s['color'], linestyle=s['linestyle'],
                marker=run_markers[run_name], markersize=6,
                linewidth=1.8, label=s['label_prefix'])
    ax.set_title(run_name, fontsize=11, fontweight='bold')
    ax.set_xlabel('Tick', fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(50, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.axvline(120, color='black', linestyle=':', alpha=0.4, linewidth=1)
    ax.text(120, 53, 'tick 120\nconverged', ha='center', va='bottom',
            fontsize=8, color='#444', alpha=0.7)

axes[0].set_ylabel('Cooperation rate', fontsize=10)
axes[-1].legend(loc='lower right', fontsize=9, framealpha=0.95)

fig.suptitle('Cooperation trajectories across N=3 replications',
             fontsize=12, fontweight='bold', y=1.02)

os.makedirs('figures', exist_ok=True)
out_png = 'figures/cooperation_trajectory.png'
out_pdf = 'figures/cooperation_trajectory.pdf'
plt.savefig(out_png, dpi=150, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
print(f'Saved: {out_png} and {out_pdf}')
