import cv2
import matplotlib.pyplot as plt

img = cv2.imread("lena_noise.jpg", cv2.IMREAD_GRAYSCALE)
K_sizes = [3, 5, 7, 11]

plt.figure(figsize=(12, 5))
plt.subplot(1, len(K_sizes) + 1, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

for idx, k in enumerate(K_sizes):
    mean = cv2.blur(img, (k, k))
    plt.subplot(1, len(K_sizes) + 1, idx + 2)
    plt.imshow(mean, cmap="gray")
    plt.title(f"Moyen {k}x{k}")
    plt.axis("off")

plt.suptitle("Filtre moyenneur")
plt.tight_layout()
plt.show()




#median
plt.figure(figsize=(12, 5))
plt.subplot(1, len(K_sizes) + 1, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

for idx, k in enumerate(K_sizes):
    median = cv2.medianBlur(img, k)
    plt.subplot(1, len(K_sizes) + 1, idx + 2)
    plt.imshow(median, cmap="gray")
    plt.title(f"Médian {k}")
    plt.axis("off")

plt.suptitle("Filtre médian")
plt.tight_layout()
plt.show()



import numpy as np

kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)
custom = cv2.filter2D(img, -1, kernel)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(custom, cmap="gray")
plt.title("Noyau personnalisé")
plt.axis("off")
plt.tight_layout()
plt.show()
