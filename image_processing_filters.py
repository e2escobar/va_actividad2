import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

IMAGE_DIR = "./images"
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']

TITLES = [
    "Filtro de Media", "Filtro Gaussiano", "Filtro de Mediana",
    "Laplaciano", "Sobel X", "Sobel Y", "Sobel Magnitud",
    "Erosión", "Dilatación", "Apertura", "Clausura", "Canny"
]

def load_images(image_dir, extensions):
    paths = [img for ext in extensions for img in glob.glob(os.path.join(image_dir, ext))]
    print(f"Se encontraron {len(paths)} imágenes")

    images = {}
    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images[os.path.basename(path)] = img
    return images

def apply_spatial_filters(images: dict) -> dict:
    spatial_results = {}
    for name, image in images.items():
        results = {
            'blur': cv2.blur(image, (5, 5)),
            'gaussian': cv2.GaussianBlur(image, (5, 5), 0),
            'median': cv2.medianBlur(image, 5),
            'laplacian': cv2.Laplacian(image, cv2.CV_64F),
            'sobel_x': cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3),
            'sobel_y': cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3),
        }
        results['sobel_mag'] = cv2.magnitude(results['sobel_x'], results['sobel_y'])
        results['canny'] = cv2.Canny(image, 100, 200)
        spatial_results[name] = results.copy() 
    return spatial_results

def apply_morphological_filters(images: dict) -> dict:
    morph_results = {}
    kernel = np.ones((5, 5), np.uint8)
    for name, image in images.items():
        results = {
            'erosion': cv2.erode(image, kernel, iterations=1),
            'dilation': cv2.dilate(image, kernel, iterations=1),
            'opening': cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel),
            'closing': cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        }
        morph_results[name] = results.copy()
    return morph_results

def show_images(image_name, original, spatial_results, morph_results, titles, save_path=None):
    images_to_show = [
        spatial_results['blur'],
        spatial_results['gaussian'],
        spatial_results['median'],
        spatial_results['laplacian'],
        spatial_results['sobel_x'],
        spatial_results['sobel_y'],
        spatial_results['sobel_mag'],
        morph_results['erosion'],
        morph_results['dilation'],
        morph_results['opening'],
        morph_results['closing'],
        spatial_results['canny']
    ]

    num_images = len(images_to_show)
    cols = 4
    rows = (num_images + cols - 1) // cols

    fig = plt.figure(figsize=(12, 10), dpi=100) # Assign to fig
    for i, (img, title) in enumerate(zip(images_to_show, titles)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray_r')
        plt.title(title, fontsize=10)
        plt.axis('off')

    plt.suptitle(f"Resultados para: {image_name}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout for suptitle
    
    if save_path:
        # save_path is just the filename, it will be saved in the current working directory.
        full_save_path = os.path.join(os.getcwd(), save_path)
        fig.savefig(full_save_path)
        print(f"Saved results for {image_name} to {full_save_path}")
    else:
        plt.show()
    plt.close(fig) # Close the figure explicitly

def main():
    parser = argparse.ArgumentParser(description="Process images with various filters and optionally save the results.")
    parser.add_argument(
        "--save", 
        action="store_true", 
        help="Save the processed image collages to the current working directory."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=IMAGE_DIR,
        help=f"Directory to load images from. Default: {IMAGE_DIR}"
    )
    args = parser.parse_args()

    images = load_images(args.image_dir, IMAGE_EXTENSIONS) 
    if not images:
        print(f"No images found in '{args.image_dir}'. Exiting.")
        return

    spatial_filtered = apply_spatial_filters(images)
    morphological_filtered = apply_morphological_filters(images)

    for name in images:
        save_filename = None
        if args.save:
            base, _ = os.path.splitext(name)
            safe_base = base.replace(" ", "_") # Sanitize filename
            save_filename = f"filtered_results_{safe_base}.png"
            
        show_images(
            image_name=name,
            original=images[name],
            spatial_results=spatial_filtered[name],
            morph_results=morphological_filtered[name],
            titles=TITLES,
            save_path=save_filename # Pass the save_filename (which might be None)
        )

if __name__ == "__main__":
    main()