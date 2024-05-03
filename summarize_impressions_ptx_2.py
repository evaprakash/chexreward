import pandas as pd
import csv
import string
import os
import re
from openai import OpenAI
from collections import Counter

file_path = 'ptx_mask_reports.csv'
data = pd.read_csv(file_path)

masks = data['mask'].tolist()
reports = data['original_report'].tolist()

reports = [re.sub(r'\s{2,}', ' ', report.replace('\n', ' ')) for report in reports]

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
responses = []

for i in range(len(reports)):
    report = reports[i]
    #prompt="Summarize the findings of the following medical report in 5 words or less in a comma-separated list:\n\"" + report + "\""
    prompt="Summarize the findings of the following medical report in 5 words or less:\n\"" + report + "\""
    print(prompt)
    out = client.chat.completions.create(messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}], model="gpt-4", temperature=0)
    response = out.choices[0].message.content.rstrip(string.punctuation)
    responses.append(response.lower())
    print("Done ", str(i))

filename = 'ptx_prompt_2.csv'

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['mask', 'prompt'])
    for m, r in zip(masks, responses):
        writer.writerow([m, r])

