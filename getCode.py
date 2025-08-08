import os
import argparse

def extract_python_code_from_directory(output_file):
    """
    Extracts Python code from all .py files in the current directory
    and saves them to a single output file.

    Args:
        output_file (str): The path to the output file.
    """

    output_file="extracted_code.txt"

    try:
        current_directory = os.getcwd()
        # Get the name of the current script to avoid it reading itself
        script_name = os.path.basename(__file__)
        all_files = os.listdir(current_directory)

        # Find all files in the current directory that end with .py
        python_files = [f for f in all_files if f.endswith('.py') and f != script_name]

        if not python_files:
            print("No Python files found in the current directory (besides this script).")
            return

        with open(output_file, 'w', encoding='utf-8') as outfile:
            for filename in python_files:
                # Write a header to the output file indicating where the code came from
                outfile.write(f"#" * 80 + "\n")
                outfile.write(f"# Code from: {filename}\n")
                outfile.write(f"#" * 80 + "\n\n")
                try:
                    with open(filename, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())
                        outfile.write('\n\n')
                except Exception as e:
                    outfile.write(f"# Error reading file {filename}: {e}\n\n")


        print(f"Successfully extracted Python code from all .py files to '{output_file}'")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    extract_python_code_from_directory("extracted_code.txt")