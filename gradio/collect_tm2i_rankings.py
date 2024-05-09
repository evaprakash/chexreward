import os
import csv

# Function to read the prompt corresponding to an image_id from tm2i_rank_order_final.csv
def get_prompt(image_id):
    # Temporarily remove '_ptx' from the end of the image_id if present
    original_image_id = image_id
    if image_id.endswith('_ptx'):
        image_id = image_id[:-4]
    with open("tm2i_rank_order_final.csv", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['image_id'] == image_id:
                return row['prompt']
    return None

# Initialize the examples list and set to track seen pairs
examples = []
seen_pairs = set()

# Iterate through each folder in tm2i_ranks
for folder_name in os.listdir("tm2i_ranks"):
    folder_path = os.path.join("tm2i_ranks", folder_name)
    
    # Check if the current item is a directory
    if os.path.isdir(folder_path):
        mask_folder = os.path.join(folder_path, "mask")
        text_folder = os.path.join(folder_path, "text")
        
        # Check if "mask" and "text" folders exist
        if os.path.isdir(mask_folder) and os.path.isdir(text_folder):
            # Iterate through each file in the mask folder
            for filename in os.listdir(mask_folder):
                if filename != "-1.txt" and filename.endswith(".txt"):
                    file_path = os.path.join(mask_folder, filename)
                    with open(file_path, 'r') as file:
                        for line in file:
                            image_id, rank_a, rank_b, rank_c, rank_d = line.strip().split(',')
                            prompt = get_prompt(image_id)
                            if prompt:
                                for i, rank_m_1 in enumerate([rank_a, rank_b, rank_c, rank_d]):
                                    for j, rank_m_2 in enumerate([rank_a, rank_b, rank_c, rank_d]):
                                        if i != j:
                                            sub_1 = str(i + 1)
                                            sub_2 = str(j + 1)
                                            image_1 = f"images/tm2i/{image_id}_{sub_1}.jpg"
                                            conditioning_image = f"images/tm2i/{image_id}_mask_{sub_1}.jpg"
                                            image_2 = f"images/tm2i/{image_id}_{sub_2}.jpg"
                                            pair_key = tuple(sorted([image_1, image_2]))  # Create a unique key for the pair
                                            if pair_key not in seen_pairs:  # Check if the pair has been seen before
                                                rank_t_1, rank_t_2 = None, None
                                                # Find rank_t_1 and rank_t_2 from the text folder
                                                text_file = os.path.join(text_folder, filename)
                                                if os.path.isfile(text_file):
                                                    with open(text_file, 'r') as text_file:
                                                        ranks = text_file.readline().strip().split(',')[1:]
                                                        rank_t_1 = ranks[i]
                                                        rank_t_2 = ranks[j]
                                                        print(ranks, rank_t_1, rank_t_2)
                                                if rank_t_1 and rank_t_2:  # Check if rank_t_1 and rank_t_2 are found
                                                    if (not(rank_t_1 == rank_t_2 and rank_m_1 == rank_m_2)):
                                                        examples.append({
                                                            "text": prompt,
                                                            "image_1": image_1,
                                                            "conditioning_image": conditioning_image,
                                                            "image_2": image_2,
                                                            "rank_t_1": rank_t_1,
                                                            "rank_m_1": rank_m_1,
                                                            "rank_t_2": rank_t_2,
                                                            "rank_m_2": rank_m_2
                                                        })
                                                    seen_pairs.add(pair_key)  # Add the pair to the set of seen pairs

# Print the examples list
for example in examples:
    print(example)

# Open a file for writing
with open("train_dpo.jsonl", "w") as f:
    # Iterate over each example
    for example in examples:
        # Write the example as a line in the file
        f.write(str(example) + "\n")
