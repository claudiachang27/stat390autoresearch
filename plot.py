import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

df = pd.read_csv('results.tsv', sep='\t')

# Color map by status
color_map = {
    'keep':     'green',
    'discard':  'red',
    'crash':    'gray',
    'baseline': 'blue',
}

fig, ax = plt.subplots(figsize=(12, 6))

# Plot running best line
best = df['val_auc'].cummax()
ax.plot(df['experiment_id'], best, color='black', linewidth=1.5, label='Running best', zorder=1)

# Plot baseline reference line
baseline_rows = df[df['status'] == 'baseline']
if not baseline_rows.empty:
    baseline_val = baseline_rows.iloc[0]['val_auc']
    ax.axhline(y=baseline_val, color='blue', linestyle='--', linewidth=1,
               label=f'Baseline ref ({baseline_val:.6f})', zorder=1)

# Plot dots colored by status
for _, row in df.iterrows():
    color = color_map.get(str(row['status']).lower(), 'gray')
    ax.scatter(row['experiment_id'], row['val_auc'], color=color, s=60, zorder=2)

# Legend
legend_handles = [
    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='keep'),
    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',   markersize=8, label='discard'),
    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',  markersize=8, label='crash'),
    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',  markersize=8, label='baseline'),
    mlines.Line2D([0], [0], color='black', linewidth=1.5, label='Running best'),
]
if not baseline_rows.empty:
    legend_handles.append(
        mlines.Line2D([0], [0], color='blue', linestyle='--', linewidth=1,
                      label=f'Baseline ref ({baseline_val:.6f})')
    )
ax.legend(handles=legend_handles, loc='lower right', fontsize=8)

ax.set_xlabel('Experiment ID')
ax.set_ylabel('val_auc (AUC-ROC)')
ax.set_title('Validation AUC-ROC Over Experiments')
ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('performance.png', dpi=150)
print('Done! performance.png generated.')
