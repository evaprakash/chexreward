import pandas as pd
import csv
import string
import os
from openai import OpenAI
from collections import Counter

def synchronize_lists(list1, list2):
    counter = Counter(list1)
    repeated_indices = {item: [i for i, x in enumerate(list1) if x == item] for item, count in counter.items() if count > 1}
    indices_to_remove = [index for indices in repeated_indices.values() for index in indices[1:]]
    indices_to_remove.sort(reverse=True)

    for index in indices_to_remove:
        del list1[index]
        del list2[index]

    return list1, list2

split_df = pd.read_csv('/deep/group/mimic-cxr/mimic-cxr-2.0.0-split.csv')
validate_split_df = split_df[split_df['split'] == 'validate'].copy()
validate_split_df['study_id'] = 's' + validate_split_df['study_id'].astype(str)

section_df = pd.read_csv('/deep/group/mimic-cxr/mimic_cxr_sectioned.csv')
matched_studies = section_df[section_df['study'].isin(validate_split_df['study_id'])]
impressions = matched_studies['impression']
impressions = impressions.tolist()
impressions = [impression.replace('\n', '') for impression in impressions if impression == impression]

matched_study_ids = matched_studies['study'].tolist()
filtered_impressions = []
filtered_matched_study_ids = []
for i in range(len(impressions)):
    impression = impressions[i]
    study_id = matched_study_ids[i]
    if impression == impression:
        filtered_impressions.append(impression.replace('\n', ''))
        filtered_matched_study_ids.append(study_id)
impressions = filtered_impressions
matched_study_ids = filtered_matched_study_ids
impressions, matched_study_ids = synchronize_lists(impressions, matched_study_ids)

df = pd.DataFrame(matched_study_ids, columns=['study_id'])
df.to_csv('study_ids.csv', index=False)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
responses = []

for i in range(len(impressions)):
    impression = impressions[i]
    #prompt="Summarize the findings of the following medical report in 5 words or less in a comma-separated list:\n\"" + impression + "\""
    prompt="Based on these chest x-ray reports, please write a five-word caption with the main finding. Don't make comparisons with previous studies, so do not use words such as \"unchanged\", \"improved\", \"worsened\", \"no change\", \"increased\", \"decreased\", etc. in the caption. Don't use commas or quotation marks in the caption. If it is normal or no problems are detected, just return Normal as the caption:\n\"" + impression + "\""
    #prompt="Summarize the findings of the following medical report in 5 words or less:\n\"" + impression + "\""
    print(prompt)
    out = client.chat.completions.create(messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}], model="gpt-4", temperature=0)
    response = out.choices[0].message.content.rstrip(string.punctuation)
    print(response)
    responses.append(response.lower())
    print("Done ", str(i))

filename = 'prompt_final.csv'

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Study ID', 'Original Impression', 'GPT-4 Summary'])
    for m, i, s in zip(matched_study_ids, impressions, responses):
        writer.writerow([m, i, s])
