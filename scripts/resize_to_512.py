from PIL import Image
import sys
import os

def resize_image(input_path, output_path, size=(512, 512)):
    img = Image.open(input_path)
    img = img.convert('RGB')  # Ensure 3 channels
    img = img.resize(size, Image.LANCZOS)
    img.save(output_path)
    print(f"Saved resized image to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python resize_to_512.py <input_image> <output_image>")
        sys.exit(1)
    input_image = sys.argv[1]
    output_image = sys.argv[2]
    resize_image(input_image, output_image)
