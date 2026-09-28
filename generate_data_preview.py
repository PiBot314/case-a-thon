import csv
from pathlib import Path
from dotenv import dotenv_values

config = dotenv_values(".env")
raw_data_folder = config["raw_data_folder"]

print(raw_data_folder)

def create_data_previews(folder_path: str):
    source_dir = Path(folder_path)
    output_dir = source_dir.parent / "data_preview"
    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in source_dir.glob("*.csv"):
        # Prevent reading files inside data_preview directory
        if csv_file.parent == output_dir:
            continue
        
        output_file = output_dir / f"preview_{csv_file.name}"
        
        with open(csv_file, mode='r', encoding='utf-8', errors='ignore') as infile, \
             open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
            
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            
            for i, row in enumerate(reader):
                if i > 5:  # Header (index 0) + first 5 lines (indices 1 to 5)
                    break
                writer.writerow(row)

if __name__ == "__main__":
    create_data_previews(raw_data_folder)