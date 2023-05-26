import csv

# Open the two CSV files
with open('annotations.csv', 'r') as file1, open('coordinates_new.csv', 'r') as file2:
    # Read the CSV data into dictionaries
    reader1 = csv.DictReader(file1, fieldnames=['1', '2', '3', '4'])
    reader2 = csv.DictReader(file2, fieldnames=['1', '5', '6', '7', '8'])

    # Create a dictionary to store the rows of file2.csv keyed by the first column
    file2_dict = {row['1']: row for row in reader2}

    # Create a list to store the merged data
    merged_data = []

    # Loop over the rows in file1.csv
    for row1 in reader1:
        # Get the matching row from file2_dict
        if 'cropped' not in row1['1']:
            row1['1'] = row1['1'].replace('.jpg', '_cropped.jpg')
            row1['4'] = row1['4'].replace('.jpg', '_cropped.jpg')

        row2 = file2_dict.get(row1['1'], None)
        if row2:
            # Merge the two rows and append to merged_data
            merged_data.append({**row1, **row2})

# Write the merged data to a new CSV file
with open('merged_annots.csv', 'w', newline='') as outfile:
    # Write the header row
    fieldnames = merged_data[0].keys()
    writer = csv.DictWriter(outfile, fieldnames)
    writer.writeheader()

    # Write the data rows
    writer.writerows(merged_data)