import os
import csv

# Function to read the prompt corresponding to an image_id from t2i_rank_order_final_alt.csv
def get_prompt(image_id):
    with open("t2i_rank_order_final_alt.csv", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['image_id'] == image_id:
                return row['prompt']
    return None

# Initialize the examples list and set to track seen pairs
examples = []
seen_pairs = set()

# Iterate through each folder in t2i_ranks
for folder_name in os.listdir("t2i_ranks"):
    folder_path = os.path.join("t2i_ranks", folder_name)
    if os.path.isdir(folder_path):
        # Iterate through each file in the folder
        for filename in os.listdir(folder_path):
            if filename != "-1.txt" and filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r') as file:
                    for line in file:
                        image_id, rank_a, rank_b, rank_c, rank_d = line.strip().split(',')
                        prompt = get_prompt(image_id)
                        if prompt:
                            for i, rank_t_1 in enumerate([rank_a, rank_b, rank_c, rank_d]):
                                for j, rank_t_2 in enumerate([rank_a, rank_b, rank_c, rank_d]):
                                    if i != j and rank_t_1 != rank_t_2:  # added condition
                                        sub_1 = str(i + 1)
                                        sub_2 = str(j + 1)
                                        image_1 = f"images/t2i/{image_id}_{sub_1}.jpg"
                                        image_2 = f"images/t2i/{image_id}_{sub_2}.jpg"
                                        pair_key = tuple(sorted([image_1, image_2]))  # Create a unique key for the pair
                                        print(rank_t_1)
                                        if pair_key not in seen_pairs:  # Check if the pair has been seen before
                                            examples.append({
                                                "text": prompt,
                                                "image_1": image_1,
                                                "conditioning_image": "N/A",
                                                "image_2": image_2,
                                                "rank_t_1": rank_t_1,
                                                "rank_m_1": "N/A",
                                                "rank_t_2": rank_t_2,
                                                "rank_m_2": "N/A"
                                            })
                                            seen_pairs.add(pair_key)  # Add the pair to the set of seen pairs

# Print the examples list
for example in examples:
    print(example)

with open("train_dpo.jsonl", "a") as f:
    # Iterate over each example
    for example in examples:
        # Write the example as a line in the file
        f.write(str(example) + "\n")

