import os
import shutil
import pydicom
from PIL import Image
import numpy as np

parent_folder = "Databases/CBIS-DDSM/Code/UnprocessImages/Images/"
output_folder = "Databases/CBIS-DDSM/Images"

parent_folder_MC = "Databases/CBIS-DDSM/Code/UnprocessImages/CropAndMasks/"
output_folder_masks = "Databases/CBIS-DDSM/GroundTrueMasks"
output_folder_crops = "Databases/CBIS-DDSM/Crops"

input_folder = "Databases/CBIS-DDSM/Images"
output_folder = "Databases/CBIS-DDSM/Images"
input_folder_masks = "Databases/CBIS-DDSM/GroundTrueMasks"
output_folder_masks = "Databases/CBIS-DDSM/GroundTrueMasks"
input_folder_crops = "Databases/CBIS-DDSM/Crops"
output_folder_crops = "Databases/CBIS-DDSM/Crops"

Images = os.listdir(parent_folder)
MCs = os.listdir(parent_folder_MC)
images = os.listdir(input_folder)

for folder in Images:
    first_folder = os.path.join(parent_folder, folder)
    second_folder = os.path.join(first_folder, os.listdir(first_folder)[0])
    third_folder = os.path.join(second_folder, os.listdir(second_folder)[0])

    folder_name = folder.split("P_")[1]
    source_file_path = os.path.join(third_folder, "1-1.dcm")

    # Define the destination file path with the desired name
    destination_file_name = folder_name + ".dcm"
    destination_file_path = os.path.join(output_folder, destination_file_name)

    # Copy the DICOM file to the destination folder with the new name
    shutil.copyfile(source_file_path, destination_file_path)

for folder_MC in MCs:
    first_folder_MC = os.path.join(parent_folder_MC, folder_MC)
    second_folder_MC = os.path.join(first_folder_MC, os.listdir(first_folder_MC)[0])
    third_folder_MC = os.path.join(second_folder_MC, os.listdir(second_folder_MC)[0])

    folder_name_MC = folder_MC.split("P_")[1]
    #test - 1.1 crop and 1.2 mask  train - 1.2 crop and 1.1 mask
    source_file_path_crop = os.path.join(third_folder_MC, "1-1.dcm")
    source_file_path_mask = os.path.join(third_folder_MC, "1-2.dcm")

    # Define the destination file path with the desired name
    destination_file_name_crop = folder_name_MC + "_crop.dcm"
    destination_file_path_crop = os.path.join(output_folder_crops, destination_file_name_crop)

    destination_file_name_mask = folder_name_MC + "_mask.dcm"
    destination_file_path_mask = os.path.join(output_folder_masks, destination_file_name_mask)

    # Copy the DICOM file to the destination folder with the new name
    shutil.copyfile(source_file_path_crop, destination_file_path_crop)
    shutil.copyfile(source_file_path_mask, destination_file_path_mask)

for image in images:
    image_name = image.split(".")[0]
    image_path_original = os.path.join(input_folder, image)
    image_path_mask= os.path.join(input_folder_masks, image_name + "_mask.dcm")
    image_path_crop = os.path.join(input_folder_crops, image_name + "_crop.dcm")
    
    image_path_original_output = os.path.join(output_folder, image_name + ".jpg")
    image_path_mask_output = os.path.join(output_folder_masks, image_name + "_mask.jpg")
    image_path_crop_output = os.path.join(output_folder_crops, image_name + "_crop.jpg")

    # read the DICOM file
    ds = pydicom.dcmread(image_path_original)
    ds_mask = pydicom.dcmread(image_path_mask)
    ds_crop = pydicom.dcmread(image_path_crop)

    # convert to RGB image
    image_original = ds.pixel_array
    image_mask = ds_mask.pixel_array
    image_crop = ds_crop.pixel_array

    # Convert the DICOM image data to an 8-bit NumPy array
    max_value = np.max(image_original)
    min_value = np.min(image_original)
    scaled_original = ((image_original - min_value) / (max_value - min_value) * 255).astype(np.uint8)

    max_value_m = np.max(image_mask)
    min_value_m= np.min(image_mask)
    scaled_mask = ((image_mask - min_value_m) / (max_value_m - min_value_m) * 255).astype(np.uint8)

    max_value_e = np.max(image_crop)
    min_value_c = np.min(image_crop)
    scaled_crop = ((image_crop - min_value_c) / (max_value - min_value_c) * 255).astype(np.uint8)

    # Create a PIL Image from the 8-bit NumPy array
    im = Image.fromarray(scaled_original)
    im_mask = Image.fromarray(scaled_mask)
    im_crop = Image.fromarray(scaled_crop)

    # save as JPG
    im.save(image_path_original_output)
    im_mask.save(image_path_mask_output)
    im_crop.save(image_path_crop_output)
