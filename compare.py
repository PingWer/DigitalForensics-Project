import cv2
import numpy as np

def calculate_pixel_similarity(image1, image2):
    
    difference = np.abs(image1.astype(np.float32) - image2.astype(np.float32))
    mean_difference = np.mean(difference)
    similarity = 1 - (mean_difference / 255)  # Normalizza tra 0 e 1
    
    print(f"Mean Difference: {mean_difference}, Similarity: {similarity}")
    
    return similarity

def find_matches(original_image, selected_portion, selected_coords, similarity_threshold):
    x1, y1, width, height = selected_coords
    match_count = 0

    selected_portion_gray = cv2.cvtColor(selected_portion, cv2.COLOR_RGB2GRAY)
    original_image_gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

    img_height, img_width = original_image_gray.shape

    for y in range(0, img_height - height + 1):
        for x in range(0, img_width - width + 1):
            block = original_image_gray[y:y + height, x:x + width]
            similarity = calculate_pixel_similarity(selected_portion_gray, block)
            if similarity >= similarity_threshold:
                match_count += 1
                print(f"Match found at ({x}, {y}) with similarity: {similarity}")

    if x1 == 0 and y1 == 0 and width == img_width and height == img_height:
        return 1

    print(f"Total Matches: {match_count}")
    return match_count-1
