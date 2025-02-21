import csv

# Read the input text file
input_file = "train_data.txt"
output_file = "train_data.csv"

# Initialize a list to store the processed data
data = []

# Open the input file and process it
with open(input_file, "r") as file:
    lines = file.readlines()
    i = 0
    while i < len(lines):
        # Extract ID, Name, Symptoms, and Treatments
        id = lines[i].strip()
        name = lines[i + 1].strip()
        symptoms = lines[i + 2].strip()
        
        # Skip the treatments line (4th column)
        i += 4  # Move to the next entry
        
        # Append the data to the list
        data.append({"ID": id, "Name": name, "Symptoms": symptoms})

# Write the processed data to a CSV file
with open(output_file, mode="w", newline="", encoding="utf-8") as csv_file:
    fieldnames = ["ID", "Name", "Symptoms"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    # Write the header
    writer.writeheader()
    
    # Write the rows
    for row in data:
        writer.writerow(row)

print(f"Data has been successfully written to {output_file}")