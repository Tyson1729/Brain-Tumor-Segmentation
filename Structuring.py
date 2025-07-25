# Importing all Libraries
import numpy as np
import nibabel as nib  # For loading .nii medical image files
import glob  # For file pattern matching
import tensorflow as tf
from tensorflow.keras.utils import to_categorical  # For one-hot encoding segmentation masks
import matplotlib.pyplot as plt  # For visualizing MRI slices
from tifffile import imwrite  # For saving multi-channel TIFF files
from sklearn.preprocessing import MinMaxScaler  # For normalizing image intensity values

scaler = MinMaxScaler()  # Initialize the MinMaxScaler
TRAIN_DATASET_PATH = r"C:\Users\sheew\Downloads\BraTS\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"

# Load flair modality and normalize
test_image_flair = nib.load(TRAIN_DATASET_PATH + '/BraTS20_Training_355/BraTS20_Training_355_flair.nii').get_fdata()
print(test_image_flair.max())  # Print max before normalization

test_image_flair = scaler.fit_transform(test_image_flair.reshape(-1, test_image_flair.shape[-1])).reshape(test_image_flair.shape)
print(test_image_flair.max())  # Print max after normalization

# Load and normalize other modalities: t1, t1ce, t2
test_image_t1 = nib.load(TRAIN_DATASET_PATH + '/BraTS20_Training_355/BraTS20_Training_355_t1.nii').get_fdata()
test_image_t1 = scaler.fit_transform(test_image_t1.reshape(-1, test_image_t1.shape[-1])).reshape(test_image_t1.shape)

test_image_t1ce = nib.load(TRAIN_DATASET_PATH + '/BraTS20_Training_355/BraTS20_Training_355_t1ce.nii').get_fdata()
test_image_t1ce = scaler.fit_transform(test_image_t1ce.reshape(-1, test_image_t1ce.shape[-1])).reshape(test_image_t1ce.shape)

test_image_t2 = nib.load(TRAIN_DATASET_PATH + '/BraTS20_Training_355/BraTS20_Training_355_t2.nii').get_fdata()
test_image_t2 = scaler.fit_transform(test_image_t2.reshape(-1, test_image_t2.shape[-1])).reshape(test_image_t2.shape)

# Load and process segmentation mask
test_mask = nib.load(TRAIN_DATASET_PATH + '/BraTS20_Training_355/BraTS20_Training_355_seg.nii').get_fdata()
test_mask = test_mask.astype(np.uint8)  # Convert to integer type
print(np.unique(test_mask))  # Print unique labels

test_mask[test_mask==4] = 3  # Replace label 4 with 3 (to keep labels in 0–3 range)
print(np.unique(test_mask))

# Visualize one random slice from each modality and mask
import random
n_slice = random.randint(0, test_mask.shape[2])
plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.imshow(test_image_flair[:,:,n_slice], cmap='gray')
plt.title('Image flair')
plt.subplot(232)
plt.imshow(test_image_t1[:,:,n_slice], cmap='gray')
plt.title('Image t1')
plt.subplot(233)
plt.imshow(test_image_t1ce[:,:,n_slice], cmap='gray')
plt.title('Image t1ce')
plt.subplot(234)
plt.imshow(test_image_t2[:,:,n_slice], cmap='gray')
plt.title('Image t2')
plt.subplot(235)
plt.imshow(test_mask[:,:,n_slice])
plt.title('Mask')
plt.show()

# Stack selected modalities and crop region of interest
combined_x = np.stack([test_image_flair, test_image_t1ce, test_image_t2], axis=3)
combined_x = combined_x[56:184, 56:184, 13:141]  # Crop for patch extraction
test_mask = test_mask[56:184, 56:184, 13:141]

# Visualize the cropped data
n_slice = random.randint(0, test_mask.shape[2])
plt.figure(figsize=(12, 8))
plt.subplot(221)
plt.imshow(combined_x[:,:,n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(222)
plt.imshow(combined_x[:,:,n_slice, 1], cmap='gray')
plt.title('Image t1ce')
plt.subplot(223)
plt.imshow(combined_x[:,:,n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(224)
plt.imshow(test_mask[:,:,n_slice])
plt.title('Mask')
plt.show()

# Save the combined image as .tif and .npy
imwrite(r"C:\Users\sheew\Downloads\BraTS\BraTS2020_TrainingData\combined355.tif", combined_x)
np.save(r"C:\Users\sheew\Downloads\BraTS\BraTS2020_TrainingData\combined355.npy", combined_x)

# Load the .npy file back and verify
my_img = np.load(r"C:\Users\sheew\Downloads\BraTS\BraTS2020_TrainingData\combined355.npy")
combined_x == my_img.all()  # Check if saved and loaded data match

# One-hot encode the mask for segmentation training
test_mask = to_categorical(test_mask, num_classes=4)

# Get sorted lists of each MRI modality and mask files
t2_list = sorted(glob.glob(r"C:\Users\sheew\Downloads\BraTS\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\*\*t2.nii"))
t1ce_list = sorted(glob.glob(r"C:\Users\sheew\Downloads\BraTS\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\*\*t1ce.nii"))
flair_list = sorted(glob.glob(r"C:\Users\sheew\Downloads\BraTS\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\*\*flair.nii"))
mask_list = sorted(glob.glob(r"C:\Users\sheew\Downloads\BraTS\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\*\*seg.nii"))

# Process all patient files
for img in range(len(t2_list)):  # All lists have same number of elements
    print("Now preparing image and masks number: ", img)
    
    # Load and normalize each modality
    temp_image_t2 = nib.load(t2_list[img]).get_fdata()
    temp_image_t2 = scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)
    
    temp_image_t1ce = nib.load(t1ce_list[img]).get_fdata()
    temp_image_t1ce = scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)
    
    temp_image_flair = nib.load(flair_list[img]).get_fdata()
    temp_image_flair = scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)
    
    # Load and process segmentation mask
    temp_mask = nib.load(mask_list[img]).get_fdata()
    temp_mask = temp_mask.astype(np.uint8)
    temp_mask[temp_mask==4] = 3  # Reassign label 4 to 3
    
    # Stack modalities into a single volume
    temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)
    
    # Crop the volume to a centered region for patch extraction
    temp_combined_images = temp_combined_images[56:184, 56:184, 13:141]
    temp_mask = temp_mask[56:184, 56:184, 13:141]
    
    # Count mask label occurrences
    val, counts = np.unique(temp_mask, return_counts=True)
    
    # Save only useful samples (at least 1% of non-background labels)
    if (1 - (counts[0]/counts.sum())) > 0.01:
        print("Save Me")
        temp_mask = to_categorical(temp_mask, num_classes=4)
        np.save(r"C:\Users\sheew\Downloads\BraTS\BraTS2020_TrainingData\Input Data\Images\image_" + str(img) + '.npy', temp_combined_images)
        np.save(r"C:\Users\sheew\Downloads\BraTS\BraTS2020_TrainingData\Input Data\Masks\mask_" + str(img) + '.npy', temp_mask)
    else:
        print("I am useless")  # Skip volumes with mostly background
