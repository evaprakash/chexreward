import json
import random

# Input and output file paths
input_file = "train_dpo_fixed.jsonl"
output_file_train = "train_t2i.jsonl"
output_file_valid = "valid_t2i.jsonl"
output_file_full = "train_t2i_full.jsonl"

# Read the input file and filter based on the condition
with open(input_file, 'r') as f:
    data = [json.loads(line) for line in f if json.loads(line)["conditioning_image"] == "N/A"]

# Shuffle the data
random.shuffle(data)

# Split the data into 90% for training and 10% for validation
split_idx = int(0.95 * len(data))
train_data = data[:split_idx]
valid_data = data[split_idx:]

# Write the full data to the full output file
with open(output_file_full, 'w') as f:
    for item in data:
        f.write(json.dumps(item) + "\n")

# Write the training data to the train output file
with open(output_file_train, 'w') as f:
    for item in train_data:
        f.write(json.dumps(item) + "\n")

# Write the validation data to the validation output file
with open(output_file_valid, 'w') as f:
    for item in valid_data:
        f.write(json.dumps(item) + "\n")

print("Processing complete.")

