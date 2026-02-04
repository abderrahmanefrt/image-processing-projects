import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 

img = cv.imread('cats.jpg')

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 3. Apply Averaging Blur
# A (3,3) kernel is very subtle. (7,7) makes the effect clearer!
average = cv.blur(img_rgb, (7,7))

# 4. Visualization
plt.figure(figsize=(12, 6))

plt.subplot(1,2,1)
plt.title("Original (RGB)")
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Averaging Blur (7x7)")
plt.imshow(average)
plt.axis('off')

plt.show()
