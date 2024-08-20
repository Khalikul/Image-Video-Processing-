from PIL import Image
import os


def crop_images_from_folder(input_folder, output_folder, crop_height, file_extensions=['.png', '.jpg', '.jpeg']):
    """
    Crop images from the top of each image in the input_folder and save them to the output_folder.

    :param input_folder: Path to the folder containing the input images.
    :param output_folder: Path to the folder where cropped images will be saved.
    :param crop_height: Height to crop from the top of each image.
    :param file_extensions: List of file extensions to consider for processing.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if any(filename.lower().endswith(ext) for ext in file_extensions):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Open the image file
            with Image.open(input_path) as img:
                # Get the size of the image
                width, height = img.size

                # Define the cropping box (left, upper, right, lower)
                box = (0, crop_height, width, height)

                # Crop the image
                cropped_img = img.crop(box)

                # Save the cropped image
                cropped_img.save(output_path)
                print(f"Cropped image saved to {output_path}")


# Example usage
input_folder = 'input_images'
output_folder = 'output_images'
crop_height = 15

crop_images_from_folder(input_folder, output_folder, crop_height)
