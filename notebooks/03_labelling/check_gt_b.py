import pandas as pd
df = pd.read_csv('data/labelled/labelled_complaints.csv', low_memory=False)
print(f'Ground Truth B (DOHMH-Matched):')
print(f'Total rows: {len(df)}')
if 'label' in df.columns:
    print(f'Severe (label=1): {int(df["label"].sum())} ({df["label"].mean():.1%})')
    print(f'Non-severe (label=0): {int((df["label"]==0).sum())} ({(df["label"]==0).mean():.1%})')
print(f'Columns: {list(df.columns)}')
print(df.head(2).to_string())
