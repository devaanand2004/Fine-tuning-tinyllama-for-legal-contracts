import pandas as pd
df = pd.read_csv('cuad_risk_labels.csv')
print('Total samples:', len(df))
print('\nRisk distribution:')
print(df['risk_level'].value_counts())
print('\nUnique clause types:', df['clause_type'].nunique())
print('\nSample counts per split (80/10/10):')
n = len(df)
print(f'  Train: {int(n*0.8)} | Val: {int(n*0.1)} | Test: {int(n*0.1)}')
