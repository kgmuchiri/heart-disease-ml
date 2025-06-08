# %%
"""
# ECG Images Preprocessing
Taking a single patients ECG image and seperates them into 12 images, one for each lead
retrieved from
"""

# %%
import os
import re
from skimage.filters import threshold_otsu, gaussian
from skimage import measure, color, morphology, filters,io
from skimage.transform import resize
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# %%
"""
## Extracting and Preprocessing Images
    
Extracts individual leads from an ECG image, preprocesses them to remove noise 
and keep only the ECG signal, and saves the smaller resulting images

Arguments:
  image_file: Name of the input image file.
  parent_folder: Path to the folder containing the image file.
  output_folder: Path to the folder where the processed lead images will be saved.
"""

# %%
def extract_and_preprocess_leads(image_file, parent_folder, output_folder):
    # Read the image
    image = imread(os.path.join(parent_folder, image_file))

    # Dividing the ECG leads with offset
    start_offset = 30 

    Lead_1 = image[300:600, 150 + start_offset:643]
    Lead_2 = image[300:600, 646 + start_offset:1135]
    Lead_3 = image[300:600, 1140 + start_offset:1626]
    Lead_4 = image[300:600, 1630 + start_offset:2125]
    Lead_5 = image[600:900, 150 + start_offset:643]
    Lead_6 = image[600:900, 646 + start_offset:1135]
    Lead_7 = image[600:900, 1140 + start_offset:1626]
    Lead_8 = image[600:900, 1630 + start_offset:2125]
    Lead_9 = image[900:1200, 150 + start_offset:643]
    Lead_10 = image[900:1200, 646 + start_offset:1135]
    Lead_11 = image[900:1200, 1140 + start_offset:1626]
    Lead_12 = image[900:1200, 1630 + start_offset:2125]

    Leads=[Lead_1,Lead_2,Lead_3,Lead_4,Lead_5,Lead_6,Lead_7,Lead_8,Lead_9,Lead_10,Lead_11,Lead_12]

     # Extract the original filename without extension
    base_filename = os.path.splitext(image_file)[0]

    # Extract the class label from the base filename
    class_label = re.match(r'([^\(]+)', base_filename).group(1) 
        # Create class subfolders within the output folder if they don't exist

    folder_name = re.sub('.jpg', '', image_file)
    output_path = os.path.join(output_folder, folder_name)

    class_output_folder = os.path.join(output_folder, class_label)
    os.makedirs(class_output_folder, exist_ok=True)

    for x, lead_img in enumerate(Leads):
        # Convert to grayscale
        grayscale = color.rgb2gray(lead_img)
        # Smooth the image
        blurred_image = gaussian(grayscale, sigma=0.7)
        # Thresholding
        global_thresh = filters.threshold_otsu(blurred_image)
        binary_global = blurred_image < global_thresh 
        # Morphological Operations (Connect broken segments)
        binary_global = morphology.closing(binary_global, morphology.square(3)) 
        # Resize
        binary_global = resize(binary_global, (180,230))
       
        # Find contours to isolate the ECG signal
        contours = measure.find_contours(binary_global, 0.8)
        contours_shape = sorted([x.shape for x in contours])[::-1][0:1]
        # Create a blank image to draw the extracted signal
        extracted_signal = np.zeros_like(binary_global)
        for contour in contours:
            if contour.shape in contours_shape:
                # Draw the contour on the blank image
                for point in contour:
                    x_coord, y_coord = int(point[1]), int(point[0])
                    extracted_signal[y_coord, x_coord] = 1  # Set pixel to white
        

        # Create the output filename with the desired convention
        output_filename = f"{base_filename}_lead_{x+1}.png"  
        output_path = os.path.join(class_output_folder, output_filename)  # Save directly to output_folder
        # Save the extracted signal image
        imsave(output_path, extracted_signal)

#only run if there are no images
'''# %%
input_folder = '../data/dataset/original_datasets/ecg_images'  # Input folder
output_folder = '../data/dataset/processed_datasets/ecg_images/trial'  # Desired output folder

# Iterate over each class folder and extract and preprocess the leads
# Uncomment to run

for class_label in ['AB', 'HMI', 'MI', 'Normal']:
    class_path = os.path.join(input_folder, class_label)
    for filename in os.listdir(class_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            extract_and_preprocess_leads(filename, class_path, output_folder)

'''

# %%
def create_datasets(folder):
    image_data = []
    labels = []
    for class_label in ['HB', 'MI', 'PMI', 'Normal']:
        class_folder = os.path.join(folder, class_label)
        for filename in os.listdir(class_folder):
            if filename.endswith('.png'):
                # Load image and convert to grayscale
                img = imread(os.path.join(class_folder, filename), as_gray=True)  
                image_data.append(img)
                labels.append(class_label)
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Print label encoding
    for i, label in enumerate(label_encoder.classes_):
        print(f"{i}: {label}")

    return np.array(image_data), encoded_labels

imagefolder= '../data/dataset/processed_datasets/ecg_images/images'
# Create and save the datasets
image_data, labels = create_datasets(imagefolder)
# Create the numpy subfolder if it doesn't exist
numpy_folder = '../data/dataset/processed_datasets/ecg_images/numpy'
os.makedirs(numpy_folder, exist_ok=True)

# Construct the full file paths within the numpy subfolder
image_data_path = os.path.join(numpy_folder, 'image_data.npy')
labels_path = os.path.join(numpy_folder, 'labels.npy')

# Save the datasets

#np.save(image_data_path, image_data)
#np.save(labels_path, labels)
# %%
image_data.shape

# %%
labels.shape

# %%
unique_labels, counts = np.unique(labels, return_counts=True)
print("Unique labels:", unique_labels)
print("Counts:", counts)

# %%
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    image_data, labels, test_size=0.2, random_state=42
)



# Construct the full file paths within the numpy subfolder
X_train_path = os.path.join(numpy_folder, 'X_train.npy')
X_test_path = os.path.join(numpy_folder, 'X_test.npy')
y_train_path = os.path.join(numpy_folder, 'y_train.npy')
y_test_path = os.path.join(numpy_folder, 'y_test.npy')

# Save the arrays
np.save(X_train_path, X_train)
np.save(X_test_path, X_test)
np.save(y_train_path, y_train)
np.save(y_test_path, y_test)

# %%
print("Min pixel value:", image_data.min())
print("Max pixel value:", image_data.max())


# %%
X_train.dtype