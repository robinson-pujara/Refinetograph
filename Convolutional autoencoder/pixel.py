import cv2
import os

def pixilate_image(image, pixel_size):
    # Get the height and width of the image
    height, width = image.shape[:2]

    # Resize the image to a smaller size
    small_image = cv2.resize(image, (width // pixel_size, height // pixel_size))

    # Resize it back to the original size
    pixilated_image = cv2.resize(small_image, (width, height), interpolation=cv2.INTER_NEAREST)

    return pixilated_image

def main():
    # Load an image
    input_image_path = 'test_img/4.jpg'
    image = cv2.imread(input_image_path)

    image = cv2.resize(image, (256, 256))

    # Set the pixel size for pixilation
    pixel_size = 10  # Adjust this value as needed


    # Pixilate the image
    pixilated_image = pixilate_image(image, pixel_size)

    # Save the pixilated image with the original name plus "_pixilated" suffix
    filename, file_extension = os.path.splitext(input_image_path)
    output_image_path = filename + '_pixilated' + file_extension
    cv2.imwrite(output_image_path, pixilated_image)

if __name__ == "__main__":
    main()

