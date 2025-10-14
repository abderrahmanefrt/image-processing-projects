import cv2
import numpy as np
import matplotlib.pyplot as plt
import random



lena = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
alex = cv2.imread('alex.png', cv2.IMREAD_GRAYSCALE)

B = np.zeros_like(lena, dtype=np.uint8)

h, w = lena.shape
x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
x2, y2 = random.randint(w//2, w-1), random.randint(h//2, h-1)


cv2.rectangle(B, (x1, y1), (x2, y2), color=255, thickness=-1)

addition = cv2.add(lena, B)
soustraction = cv2.subtract(lena, B)
multiplication = cv2.multiply(lena, B, scale=1/255)


And=cv2.bitwise_and(lena,B)
Or=cv2.bitwise_or(lena,B)
Xor=cv2.bitwise_xor(lena,B)


plt.subplot(2, 4, 8)
plt.imshow(lena, cmap='gray')
plt.title("Lena originale")
plt.axis('off')


plt.subplot(2, 4, 2)
plt.imshow(B, cmap='gray')
plt.title("Image binaire B")
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(addition, cmap='gray')
plt.title("Addition lena + B")
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(soustraction, cmap='gray')
plt.title("Soustraction lena - B")
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(multiplication, cmap='gray')
plt.title("Multiplication lena * B")
plt.axis('off')

plt.subplot(2, 4, 5)
plt.imshow(And, cmap='gray')
plt.title("And lena * B")
plt.axis('off')


plt.subplot(2, 4, 6)
plt.imshow(Or, cmap='gray')
plt.title("Or lena * B")
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(Xor, cmap='gray')
plt.title("Xor lena * B")
plt.axis('off')

plt.tight_layout()
plt.show()
