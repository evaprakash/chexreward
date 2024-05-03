import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
'''
prompt= "Summarize the findings of the following medical report in 5 words or less:\n Clinical data: Left pneumothorax. Drain stopped bubbling. ? residual pneumothorax.; Medical question: Chest ([ADDRESS]) Comparison: Chest X-ray (PA) dated [DATE]. Mild reduction is seen for the small left apical pneumothorax to about half of what it was (distance from apex about 0.6cm). The tip of the left chest drain is seen inferior to this. The small amount of left chest wall surgical emphysema has largely resolved. The small amount of fluid in the left pleural space has reduced. The rest of both lungs and right pleural space remain clear. Transcribed by: dh Dr. [PERSONALNAME] [PERSONALNAME]"
'''

prompt="Based on these chest x-ray reports, please write a five-word caption with the main finding. Don't make comparisons with previous studies. Don't use commas. If it is normal, just return \"Normal\":\nClinical data: Left pneumothorax. Drain stopped bubbling. ? residual pneumothorax.; Medical question: Chest ([ADDRESS]) Comparison: Chest X-ray (PA) dated [DATE]. Mild reduction is seen for the small left apical pneumothorax to about half of what it was (distance from apex about 0.6cm). The tip of the left chest drain is seen inferior to this. The small amount of left chest wall surgical emphysema has largely resolved. The small amount of fluid in the left pleural space has reduced. The rest of both lungs and right pleural space remain clear. Transcribed by: dh Dr. [PERSONALNAME] [PERSONALNAME]"

response = client.chat.completions.create(messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}], model="gpt-4",)

print(response.choices[0].message.content)


