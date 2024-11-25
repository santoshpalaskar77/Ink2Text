import numpy as np
from pdf2image import convert_from_path

def trim_pdf_image_to_png(img, output_path):
    # Convert the PDF to images
    images = convert_from_path(img)

    # Select the first (and only) image
    image = images[0]

    # Convert the image to an array
    image_array = np.array(image)

    # Convert to grayscale and create a binary mask (where white pixels are True)
    gray_image = np.mean(image_array, axis=2)  # Convert to grayscale
    binary_mask = gray_image > 240  # Adjust the threshold as needed for white

    # Get the coordinates of non-white pixels
    coords = np.argwhere(~binary_mask)

    # Check if any non-white pixels were found
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)

        # Crop the image using the coordinates
        trimmed_image = image.crop((x0, y0, x1 + 1, y1 + 1))
    else:
        trimmed_image = image  # No cropping needed

    # Save the trimmed image as a PNG file
    trimmed_image.save(output_path, format='PNG')
    return output_path

