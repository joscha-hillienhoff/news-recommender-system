input_file = "cbf.txt"
output_file = "cbf_sorted.txt"

# Read and parse the file
with open(input_file, "r") as f:
    lines = f.readlines()

# Parse each line into (id, vector) tuples
parsed_lines = []
for line in lines:
    id_str, vector_str = line.strip().split(" ", 1)
    parsed_lines.append((int(id_str), vector_str))

# Sort by ID
parsed_lines.sort(key=lambda x: x[0])

# Write to new file
with open(output_file, "w") as f:
    for id_num, vector_str in parsed_lines:
        f.write(f"{id_num} {vector_str}\n")

print(f"Sorted lines written to {output_file}")
