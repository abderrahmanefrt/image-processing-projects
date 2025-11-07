import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("lena_noise.jpg", cv2.IMREAD_GRAYSCALE)

kernel = np.array([[1, 0, -1],
                   [0, 1, 0],
                   [1, 0, -1]], dtype=np.float64)

def convolution_with_padding(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape

    kernel = np.flipud(np.fliplr(kernel))

    p = (kh - 1) // 2

    padded = np.pad(img, ((p, p), (p, p)), mode='constant', constant_values=0)

    output = np.zeros((h, w), dtype=np.float64)

  
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)

    return output

result = convolution_with_padding(img, kernel)

result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
result = np.uint8(result)


plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.title("Convolution avec padding")
plt.imshow(result, cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()


