import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results.tsv', sep='\t')

plt.figure(figsize=(8, 5))
plt.plot(df['experiment_id'], df['val_auc'], marker='o')
plt.xlabel('Experiment')
plt.ylabel('AUC-ROC')
plt.title('Model Performance Over Time')
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.savefig('performance.png')
print('Done! performance.png generated.')
