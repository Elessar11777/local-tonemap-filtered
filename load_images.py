import glob
import os
import cv2
import numpy as np

# Function to load images and their exposure times from a directory
def load_images(dir, img_ext):
    # Use iglob to generate an iterator of filenames with the specified extension
    iter_items = glob.iglob(dir + img_ext)

    # Initialize lists to store the images and their exposure times
    images = []
    exposures = []

    # Iterate through the filenames
    for item in iter_items:
        # Read the image and convert it from BGR to RGB color space
        images.append(cv2.cvtColor(cv2.imread(item), code=cv2.COLOR_BGR2RGB))
        # Get the image's filename without the path
        fname = os.path.basename(item)
        # Extract the exposure time from the filename (assumes the exposure time is in the format "num_1000.ext")
        num = int(fname[:-4].split('_')[-1])
        den = 1000
        exposures.append(int(num / den))

    # Create a list of tuples, where each tuple contains an image and its exposure time
    iter_tuple = zip(images, exposures)
    # Sort the list of tuples in descending order based on exposure times
    sorted_iter_tuple = sorted(iter_tuple, key=lambda pair: pair[1], reverse=True)
    # Extract the sorted images and exposure times into separate lists
    sorted_images = [img for img, times in sorted_iter_tuple]
    sorted_exposures = sorted(exposures, reverse=True)
    # Calculate the log of the sorted exposure times
    sorted_log_exposures = np.log(sorted_exposures)

    # Return the sorted images and their log exposure times as a list
    return [sorted_images, sorted_log_exposures]