import pandas as pd
df_a = pd.read_csv('data/labelled/labelled_complaints_ground_truth.csv', low_memory=False, nrows=3)
print("GT-A columns:", list(df_a.columns))
print("\nGT-A sample (key cols):")
key = [c for c in ['descriptor','descriptor_2','latitude','longitude','created_date','borough',
                    'incident_zip','hazard_tier','priority_label','complaint_text',
                    'unique_key','incident_address'] if c in df_a.columns]
print(df_a[key].to_string())
print("\nNull counts for geo cols:")
df_full = pd.read_csv('data/labelled/labelled_complaints_ground_truth.csv', low_memory=False)
for c in ['latitude','longitude','created_date','borough']:
    if c in df_full.columns:
        print(f"  {c}: {df_full[c].isna().sum()} nulls / {len(df_full)} total")
print(f"\nTotal rows: {len(df_full)}")
print("priority_label distribution:", df_full['priority_label'].value_counts().to_dict())
print("hazard_tier distribution:", df_full['hazard_tier'].value_counts().to_dict())
