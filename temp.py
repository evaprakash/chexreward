import pandas as pd

prompt_df = pd.read_csv('prompt_1.csv')
prompts = prompt_df['Original Impression'].tolist()
print(prompts[:5])
print(prompt_df.iloc[:, 0].tolist())
