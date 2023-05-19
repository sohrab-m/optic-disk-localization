import os
import csv
from PIL import Image
from tqdm import tqdm

# Define the directory to search for images
directory = "dataset"

# Define the threshold value for brightness level
brightness_threshold = 50

# Initialize the progress bar with the total number of images
total_images = 0

# Loop over every subdirectory
for root, dirs, files in os.walk(directory):
    for file in files:
        # Check if the file is a JPEG image and does not already include "cropped" in its filename
        if file.lower().endswith(".jpg") and "cropped" not in file.lower():
            # Increment the total number of images
            total_images += 1

# Re-initialize the progress bar with the total number of images
progress_bar = tqdm(total=total_images)

# Initialize the CSV file to store the coordinates
csv_file = open('coordinates_new.csv', mode='w')
csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
csv_writer.writerow(['Filename', 'Left', 'Upper', 'Right', 'Lower'])

# Loop over every subdirectory
for root, dirs, files in os.walk(directory):
    for file in files:
        # Check if the file is a JPEG image and does not already include "cropped" in its filename
        if file.lower().endswith(".jpg") and "cropped" not in file.lower():
            # Construct the full path to the image file
            filepath = os.path.join(root, file)
            # Load the image
            image = Image.open(filepath)
            # Find the bounding box of the non-black region
            bbox = image.convert('L').point(lambda x: 0 if x < brightness_threshold else 1, mode='1').getbbox()
            # Crop the image to the bounding box
            cropped = image.crop(bbox)
            # Construct the new filename with "_cropped" added
            new_filename = os.path.splitext(file)[0] + "_cropped.jpg"
            # Save the cropped image with the new filename
            cropped.save(os.path.join(root, new_filename))
            # Write the coordinates to the CSV file
            csv_writer.writerow([new_filename, bbox[0], bbox[1], bbox[2], bbox[3]])
            # Update the progress bar
            progress_bar.update(1)

# Close the CSV file
csv_file.close()

# Close the progress bar when done
progress_bar.close()