import os

# Initialize the examples list and set to track seen pairs
examples = []
seen_pairs = set()

# Iterate through each folder in m2i_ranks
for folder_name in os.listdir("m2i_ranks"):
    folder_path = os.path.join("m2i_ranks", folder_name)
    if os.path.isdir(folder_path):
        # Iterate through each file in the folder
        for filename in os.listdir(folder_path):
            if filename != "-1.txt" and filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r') as file:
                    for line in file:
                        image_id, rank_a, rank_b, rank_c, rank_d = line.strip().split(',')
                        for i, rank_m_1 in enumerate([rank_a, rank_b, rank_c, rank_d]):
                            for j, rank_m_2 in enumerate([rank_a, rank_b, rank_c, rank_d]):
                                if i != j and rank_m_1 != rank_m_2:  # added condition
                                    sub_1 = str(i + 1)
                                    sub_2 = str(j + 1)
                                    image_1 = f"images/m2i/{image_id}_{sub_1}.jpg"
                                    conditioning_image = f"images/m2i/{image_id}_mask_{sub_1}.jpg"
                                    image_2 = f"images/m2i/{image_id}_{sub_2}.jpg"
                                    pair_key = tuple(sorted([image_1, image_2]))  # Create a unique key for the pair
                                    if pair_key not in seen_pairs:  # Check if the pair has been seen before
                                        examples.append({
                                            "text": "N/A",
                                            "image_1": image_1,
                                            "conditioning_image": conditioning_image,
                                            "image_2": image_2,
                                            "rank_t_1": "N/A",
                                            "rank_m_1": rank_m_1,
                                            "rank_t_2": "N/A",
                                            "rank_m_2": rank_m_2
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

