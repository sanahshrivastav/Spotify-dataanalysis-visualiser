import os

def remove_duplicates(file_path):
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Remove duplicates and empty lines
    unique_lines = sorted(set(line.strip() for line in lines if line.strip()))

    with open(file_path, 'w') as file:
        for line in unique_lines:
            file.write(line + '\n')

if __name__ == "__main__":
    remove_duplicates('requirements.txt')
