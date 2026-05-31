import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

df = pd.read_csv('results.tsv', sep='\t')

# Primary baseline: exp 3 (logistic regression, first clean feature set)
baseline_row = df[df['status'] == 'baseline'].iloc[0]
baseline_val = baseline_row['val_auc']   # 0.7039

# Plot all experiments from exp 3 onward
df_plot = df[df['experiment_id'] >= baseline_row['experiment_id']].copy()

# Experiments 4-38 are on the partially-leaky feature set — mark differently
LEAKY_RANGE = range(4, 39)

fig, ax = plt.subplots(figsize=(13, 6))

# Shaded region for partially-leaky experiments
ax.axvspan(3.5, 38.5, color='lightyellow', alpha=0.6, zorder=0,
           label='Partially-leaky feature set (exps 4–38)')

# Vertical line marking second leakage audit
ax.axvline(x=39, color='orange', linestyle='--', linewidth=1.2,
           label='Second leakage audit (exp 39)', zorder=1)

# Primary baseline reference
ax.axhline(y=baseline_val, color='blue', linestyle='--', linewidth=1.2,
           label=f'LogReg baseline — exp 3 ({baseline_val:.4f})', zorder=1)

# Connecting line through all plotted points
ax.plot(df_plot['experiment_id'], df_plot['val_auc'],
        color='black', linewidth=0.8, zorder=2, alpha=0.5)

# Colored dots — grey out leaky-phase experiments
for _, row in df_plot.iterrows():
    eid = row['experiment_id']
    status = str(row['status']).lower()
    if eid in LEAKY_RANGE:
        color = '#aaaaaa'   # grey for leaky phase
    elif status == 'keep':
        color = 'green'
    elif status == 'baseline':
        color = 'blue'
    elif status == 'discard':
        color = 'red'
    else:
        color = 'gray'
    ax.scatter(eid, row['val_auc'], color=color, s=70, zorder=3)

# Legend
legend_handles = [
    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                  markersize=8, label='Baseline'),
    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                  markersize=8, label='Keep (clean)'),
    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                  markersize=8, label='Discard (clean)'),
    mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#aaaaaa',
                  markersize=8, label='Exps 4–38 (leaky, invalidated)'),
    mpatches.Patch(color='lightyellow', label='Partially-leaky feature set'),
    mlines.Line2D([0], [0], color='orange', linestyle='--', linewidth=1.2,
                  label='Second leakage audit (exp 39)'),
    mlines.Line2D([0], [0], color='blue', linestyle='--', linewidth=1.2,
                  label=f'LogReg baseline — exp 3 ({baseline_val:.4f})'),
]
ax.legend(handles=legend_handles, loc='upper right', fontsize=8)

ax.set_xlabel('Experiment ID')
ax.set_ylabel('Validation AUC-ROC')
ax.set_title('Validation AUC-ROC Over 52 Experiments')
ax.grid(True, linestyle=':', alpha=0.4)
ax.set_ylim(0.68, 1.01)

plt.tight_layout()
plt.savefig('performance.png', dpi=150)
print('Done! performance.png generated.')
