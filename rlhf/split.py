import json
import random

# Path to the input JSONL file
input_file = 'train_dpo_fixed.jsonl'
# Paths to the output JSONL files
train_file = 'train.jsonl'
validation_file = 'validation.jsonl'

# Read the input JSONL file
with open(input_file, 'r') as f:
    lines = f.readlines()

# Shuffle the lines
random.shuffle(lines)

# Split the lines into validation and training sets
validation_lines = lines[:50]
train_lines = lines[50:]

# Write the validation lines to the validation JSONL file
with open(validation_file, 'w') as f:
    for line in validation_lines:
        f.write(line)

# Write the train lines to the train JSONL file
with open(train_file, 'w') as f:
    for line in train_lines:
        f.write(line)

print(f'Validation file saved to {validation_file}')
print(f'Training file saved to {train_file}')

