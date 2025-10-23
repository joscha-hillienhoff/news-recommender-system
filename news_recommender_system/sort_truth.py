"""
Utility script to sort truth files by impression ID.

This script reads a truth file, sorts the lines by their numeric ID, and writes
the sorted result to a new file. Used during preprocessing for evaluation.
"""


def sort_truth_file(input_file: str, output_file: str) -> None:
    """
    Sort a truth file by impression ID (integer) and write the result to a new file.

    Each line in the file should follow the format:
        <impression_id> <vector or json_string>

    Example:
        42 ["item1", "item2", "item3"]

    Parameters
    ----------
    input_file : str
        Path to the unsorted truth file.
    output_file : str
        Path to save the sorted file.
    """
    # Read all lines from the input file
    with open(input_file, "r") as f:
        lines = f.readlines()

    # Parse each line into (id, vector) tuples
    parsed_lines = []
    for line in lines:
        id_str, vector_str = line.strip().split(" ", 1)
        parsed_lines.append((int(id_str), vector_str))

    # Sort the list of tuples by the numeric ID
    parsed_lines.sort(key=lambda x: x[0])

    # Write the sorted lines to the output file
    with open(output_file, "w") as f:
        for id_num, vector_str in parsed_lines:
            f.write(f"{id_num} {vector_str}\n")

    print(f"✅ Sorted lines written to {output_file}")
