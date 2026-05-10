import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

df = pd.read_csv('results.tsv', sep='\t')

# Use only the LAST baseline as the true baseline reference
baseline_rows = df[df['status'] == 'baseline']
true_baseline = baseline_rows.iloc[-1]  # last baseline = leakage-free one
baseline_val = true_baseline['val_auc']
baseline_id = true_baseline['experiment_id']

# Only plot from the true baseline onward
df_plot = df[df['experiment_id'] >= baseline_id].copy()

# Color map by status
color_map = {
    'keep':     'green',
    'discard':  'red',
    'crash':    'gray',
    'baseline': 'blue',
}

fig, ax = plt.subplots(figsize=(12, 6))

# Connecting line through all plotted points
ax.plot(df_plot['experiment_id'], df_plot['val_auc'],
        color='black', linewidth=1.2, zorder=1, label='Running best')

# Baseline reference horizontal line
ax.axhline(y=baseline_val, color='blue', linestyle='--', linewidth=1,
           label=f'Baseline ref ({baseline_val:.6f})', zorder=1)

# Colored dots by status
for _, row in df_plot.iterrows():
    color = color_map.get(str(row['status']).lower(), 'gray')
    ax.scatter(row['experiment_id'], row['val_auc'], color=color, s=80, zorder=2)

# Legend
legend_handles = [
    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='keep'),
    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',   markersize=8, label='discard'),
    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',  markersize=8, label='crash'),
    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',  markersize=8, label='baseline'),
    mlines.Line2D([0], [0], color='black', linewidth=1.2, label='Running best'),
    mlines.Line2D([0], [0], color='blue', linestyle='--', linewidth=1,
                  label=f'Baseline ref ({baseline_val:.6f})'),
]
ax.legend(handles=legend_handles, loc='lower right', fontsize=8)

ax.set_xlabel('Experiment ID')
ax.set_ylabel('val_auc (AUC-ROC)')
ax.set_title('Validation AUC-ROC Over Experiments')
ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('performance.png', dpi=150)
print('Done! performance.png generated.')
