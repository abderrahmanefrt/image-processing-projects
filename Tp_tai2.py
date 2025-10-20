import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

lena = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
alex = cv2.imread('alex.png', cv2.IMREAD_GRAYSCALE)

B = np.zeros_like(lena, dtype=np.uint8)
h, w = lena.shape
x1, y1 = random.randint(0, w // 2), random.randint(0, h // 2)
x2, y2 = random.randint(w // 2, w - 1), random.randint(h // 2, h - 1)
cv2.rectangle(B, (x1, y1), (x2, y2), color=255, thickness=-1)

addition = cv2.add(lena, B)
soustraction = cv2.subtract(lena, B)
multiplication = cv2.multiply(lena, B, scale=1/255)

And = cv2.bitwise_and(lena, B)
Or = cv2.bitwise_or(lena, B)
Xor = cv2.bitwise_xor(lena, B)



def HISTO(img):
    h, w = img.shape
    hist = np.zeros(256, dtype=int)
    for y in range(h):
        for x in range(w):
            hist[img[y, x]] += 1
    return hist

hist_lena = HISTO(lena)


def TRL(img, C):
    img_trl = img.astype(np.int16) + C  # éviter débordement
    img_trl = np.clip(img_trl, 0, 255)  # rester entre 0 et 255
    return img_trl.astype(np.uint8)

lena_plus = TRL(lena, 50)
lena_minus = TRL(lena, -50)

plt.figure(figsize=(12,6))

plt.subplot(2,3,1)
plt.imshow(lena, cmap='gray')
plt.title("Image originale")
plt.axis('off')

plt.subplot(2,3,2)
plt.imshow(lena_plus, cmap='gray')
plt.title("C = +50 ")
plt.axis('off')

plt.subplot(2,3,3)
plt.imshow(lena_minus, cmap='gray')
plt.title("C = -50")
plt.axis('off')

plt.subplot(2,3,4)
plt.plot(HISTO(lena), color='black')
plt.title("Histogramme original")

plt.subplot(2,3,5)
plt.plot(HISTO(lena_plus), color='black')
plt.title("Histogramme +50")

plt.subplot(2,3,6)
plt.plot(HISTO(lena_minus), color='black')
plt.title("Histogramme -50")

plt.tight_layout()
plt.show()




plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(lena, cmap='gray')
plt.title("Image originale (Lena)")
plt.axis('off')

plt.subplot(1,2,2)
plt.plot(hist_lena, color='black')
plt.title("Histogramme de Lena")
plt.xlabel("Niveaux de gris")
plt.ylabel("Fréquence")
plt.tight_layout()
plt.show()





'''plt.figure(figsize=(12, 6))

plt.subplot(2, 4, 1)
plt.imshow(lena, cmap='gray')
plt.title("Lena originale")
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(B, cmap='gray')
plt.title("Image binaire B")
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(addition, cmap='gray')
plt.title("Addition (Lena + B)")
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(soustraction, cmap='gray')
plt.title("Soustraction (Lena - B)")
plt.axis('off')

plt.subplot(2, 4, 5)
plt.imshow(multiplication, cmap='gray')
plt.title("Multiplication (Lena * B)")
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(And, cmap='gray')
plt.title("AND (Lena & B)")
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(Or, cmap='gray')
plt.title("OR (Lena | B)")
plt.axis('off')

plt.subplot(2, 4, 8)
plt.imshow(Xor, cmap='gray')
plt.title("XOR (Lena ^ B)")
plt.axis('off')
'''

