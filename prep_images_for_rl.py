import os
import cv2
import csv
import argparse
'''
There is a parser to get the path to the folder with the images, referred to as "given path"

This script will first open folders in a "given path" and picks .jpg the files wihtin
that folder that match the folder name. These will be our training images.
Each image will be showed to the user and the user will be asked to draw a circle with opencv
around the optic disk. The center of the circle will be saved as the optic disk coordinates.

we press n to go next image until we are done with all the images in the folder.
then the relative path of each image wrt the "given path" will be saved in a csv file along with the optic disk coordinates.
The csv file will be saved to the "given path" as well.
'''
def get_image_files(folder_path):
    image_files = []
    for file in os.listdir(folder_path):
        if file.endswith(".png"):
            image_files.append(file)
    return image_files



def save_coordinates_to_csv(csv_path, image_path, coordinates):
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Image Path', 'Optic Disk Coordinates'])

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([image_path, coordinates])


#this function will be called whenever the mouse is right-clicked
def mouse_callback(event, x, y, flags, params):
    # print('image', params[1].shape)
    #right-click event value is 2
    if event == 2:
        print(x, y)
        radius = 100
        color = (0, 0, 255)
        thickness = 2
        cv2.circle(params[1], (x, y), radius, color, thickness)
        cv2.imshow(str(params[0])+'closing soon...', params[1])
        cv2.waitKey(3000)
        cv2.destroyWindow(str(params[0])+'closing soon...')
        save_coordinates_to_csv(params[2], params[3], [x, y])



def mouse_click(name, image, csv_path, image_path):
    scale=0.8
    window_width = int(image.shape[1] * scale)
    window_height = int(image.shape[0] * scale)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, window_width, window_height)

    #set mouse callback function for window
    cv2.setMouseCallback(name, mouse_callback, [name, image, csv_path, image_path])

def main(given_path): #i.e. given_path = data/sample_optic_disc
    image_folders = os.listdir(given_path) #i.e. ['14', '11', '1', '30', '4', '22', '18', '10', '2', '15', '7', '24', '16', '5', '28', '26', '20', '25', '23', '9', '12', '21', '8', '13', '3', '17', '6', '29', '19', '27']
    exit=False
    csv_path = os.path.join(given_path, "path_xy_annots_" + ".csv")

    for folder in image_folders: 
        folder_path = os.path.join(given_path, folder)

        image_path = folder_path + "/"+ folder + ".png"
        print('image path', image_path)
        optic_disk_coordinates = []
        image = cv2.imread(image_path)

        while True:
            
            mouse_click(str(folder), image, csv_path, image_path)


            cv2.imshow(str(folder), image)
            key = cv2.waitKey(0)
            if key == ord('q') or key == 27:  # 'q' or 'esc' key to quit
                exit=True
                break
            elif key != -1:  # any other key to go to the next frame
                break

        cv2.destroyAllWindows()
        if exit:
            break


def cleanup(given_path):
    '''
    method goes through the csv file that was created by the save_coordinates_to_csv() method
    and if there are any duplicates in the first row removes all by the last one'''
    csv_path = os.path.join(given_path, "path_xy_annots" + ".csv")
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        seen = set()
        for row in rows:
            if row[0] in seen:
                rows.remove(row)
                print('removed', row[0], '....duplicate found')
            else:
                seen.add(row[0])
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prep Images for RL')
    parser.add_argument('given_path', type=str, help='Path to the folder with images')
    args = parser.parse_args()
    main(args.given_path)
    cleanup(args.given_path)
