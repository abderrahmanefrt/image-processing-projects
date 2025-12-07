import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION / FUNCTIONS ---

def quantifiquation(image, levels):
    """
    Quantizes an 8-bit grayscale image into K intensity levels.

    Args:
        image (np.array): The original grayscale image (0-255).
        levels (int): The desired number of quantization levels (K).

    Returns:
        np.array: The quantized image (uses the start of the interval as the value).
    """
    # 256 is the total number of possible grayscale values for 8-bit depth.
    step = 256 // levels
    
    # Quantization: (interval index) * interval size
    # Example: If levels=4, step=64. Pixel 150 -> (150//64) * 64 = 2 * 64 = 128
    quantised_image = (image // step) * step
    return quantised_image.astype(np.uint8) # Ensure the output is 8-bit integer

# --- I. IMAGE LOADING AND INSPECTION ---

# Load image in grayscale format
image_gray = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
if image_gray is None:
    print("Error: Could not load 'lena.jpg'. Ensure the file path is correct.")
    exit()

print("--- SHAPE INSPECTION ---")
# Grayscale Shape: (Height, Width)
print(f"Grayscale Image Shape: {image_gray.shape} (2D - 1 Luminosity Channel)")
cv2.imshow('1. Grayscale Image', image_gray)


# Load image in color format (BGR is default in OpenCV)
image_color = cv2.imread('lena.jpg', cv2.IMREAD_COLOR)
# Color Shape: (Height, Width, Channels)
print(f"Color Image Shape: {image_color.shape} (3D - 3 BGR Channels)")
cv2.imshow('2. Color Image (BGR Default)', image_color)

# --- II. COLOR MANAGEMENT (RGB vs BGR and Channel Splitting) ---

# Convert BGR -> RGB for correct display with Matplotlib (which expects RGB)
image_rgb = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)

# Note: The following line from the original code would overwrite image_rgb with a grayscale version.
# Since we already have image_gray, we'll keep the RGB version for channel splitting.
# image_rgb=cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY) 

# Split the RGB image into its B, G, and R components.
# Each component becomes a separate 2D grayscale array representing that color's intensity.
b, g, r = cv2.split(image_rgb)

plt.figure(figsize=(12, 4)) # Set figure size for better visualization of 3 subplots

# Red Channel
plt.subplot(1, 3, 1)
plt.imshow(r, cmap='gray') # Displayed as grayscale as it's an intensity map
plt.title('3. Red Channel Intensity (R)')
plt.axis('off')

# Green Channel
plt.subplot(1, 3, 2)
plt.imshow(g, cmap='gray')
plt.title('4. Green Channel Intensity (G)')
plt.axis('off')

# Blue Channel
plt.subplot(1, 3, 3)
plt.imshow(b, cmap='gray')
plt.title('5. Blue Channel Intensity (B)')
plt.axis('off')

# --- III. SAMPLING (SPATIAL RESOLUTION) ---

# Reduce spatial resolution to 50x50 pixels (Downsampling)
# INTER_NEAREST is used to simulate simple pixel removal, showing the loss of detail
Img_echantillon = cv2.resize(image_gray, (50, 50), interpolation=cv2.INTER_NEAREST)

# Upsample the image back to original size to better visualize the resulting pixelation
Img_agrandie = cv2.resize(Img_echantillon, image_gray.shape[::-1], interpolation=cv2.INTER_NEAREST)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(Img_echantillon, cmap='gray')
plt.title(f'6. Downsampled ({Img_echantillon.shape[0]}x{Img_echantillon.shape[1]})')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(Img_agrandie, cmap='gray')
plt.title('7. Resized (Shows Spatial Loss)')
plt.axis('off')

# --- IV. QUANTIZATION (INTENSITY RESOLUTION) ---

# Apply the quantization function with K = 8 levels
K_levels = 8
img_quantified = quantifiquation(image_gray, K_levels)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image_gray, cmap='gray')
plt.title('8. Original (256 Levels)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_quantified, cmap='gray')
plt.title(f'9. Quantized ({K_levels} Levels)')
plt.axis('off')


# --- V. FINAL DISPLAY AND CLEANUP ---

plt.tight_layout() # Adjust space between subplots
plt.show() # Display all Matplotlib figures

# Keep the OpenCV windows open until a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
