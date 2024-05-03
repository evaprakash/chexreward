from transformers import AutoTokenizer, PretrainedConfig, AutoModelForMaskedLM, BertForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("StanfordAIMI/RadBERT")
model = AutoModelForMaskedLM.from_pretrained("StanfordAIMI/RadBERT", output_hidden_states=True)
inputs = tokenizer("pneumothorax x-ray", padding="max_length", truncation=True, max_length=77, return_tensors="pt")
with open("temp_radbert.txt", "w") as fp:
    fp.write(str(model))
print(inputs)
outputs = model(**inputs)
print(len(outputs[1]))
#outputs = outputs[-1][-1]
#outputs = torch.mean(outputs, dim=1)
print(outputs.shape)
