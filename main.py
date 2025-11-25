import cv2
import pytesseract
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # For Google Colab

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    return gray, thresh

def extract_text(image):
    text = pytesseract.image_to_string(image)
    return text

def main():
    path = input("Enter image path: ")
    img = cv2.imread(path)

    gray, thresh = preprocess_image(img)
    text = extract_text(gray)

    print("\nExtracted Text:\n", text)

    plt.figure(figsize=(12,5))
    plt.subplot(1,3,1)
    plt.title('Original')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.subplot(1,3,2)
    plt.title('Gray')
    plt.imshow(gray, cmap='gray')

    plt.subplot(1,3,3)
    plt.title('Threshold')
    plt.imshow(thresh, cmap='gray')
    plt.show()

    # Save results
    with open("results/extracted_text.txt", "w") as f:
        f.write(text)

if __name__ == "__main__":
    main()
