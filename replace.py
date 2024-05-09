import json

def replace_single_quotes_with_double_quotes(input_file, output_file):
    with open(input_file, 'r') as f:
        data = f.read()

    # Replace single quotes with double quotes
    data = data.replace("'", '"')

    # Write modified data to a new file
    with open(output_file, 'w') as f:
        f.write(data)

# Replace single quotes with double quotes in JSON file
input_file_path = '/home/eprakash/diffusers/examples/controlnet/train_data/train_dpo.jsonl'
output_file_path = '/home/eprakash/diffusers/examples/controlnet/train_data/train_dpo_fixed.jsonl'

replace_single_quotes_with_double_quotes(input_file_path, output_file_path)

