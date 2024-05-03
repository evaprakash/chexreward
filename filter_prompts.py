import pandas as pd

column_name = 'prompt'
df = pd.read_csv('ptx_prompt_final.csv')
filtered_df = df[df[column_name] != 'normal']
filtered_df.to_csv('ptx_prompt_final_filtered.csv', index=False)
