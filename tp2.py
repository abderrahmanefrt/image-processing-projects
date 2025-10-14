# ============================================================
# TAI_TP2.py
# Traitement et Analyse d’Image – TP2
# Opérations arithmétiques et logiques sur les images
# ============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# ------------------------------------------------------------
# 1. Charger les images en niveaux de gris
# ------------------------------------------------------------
lena = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
alex = cv2.imread('alex.png', cv2.IMREAD_GRAYSCALE)

if lena is None or alex is None:
    print("Erreur : Vérifiez que 'lena.jpg' et 'alex.png' sont dans le même dossier que ce script.")
    exit()

# Redimensionner alex à la même taille que lena (si nécessaire)
alex = cv2.resize(alex, lena.shape[::-1])

# ------------------------------------------------------------
# 2. Créer une image binaire B avec un rectangle aléatoire
# ------------------------------------------------------------
# Créer une image noire (même taille que lena)
B = np.zeros_like(lena, dtype=np.uint8)

# Générer des coordonnées aléatoires du rectangle
h, w = lena.shape
x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
x2, y2 = random.randint(w//2, w-1), random.randint(h//2, h-1)

# Dessiner le rectangle blanc dans l’image B
cv2.rectangle(B, (x1, y1), (x2, y2), color=255, thickness=-1)

# ------------------------------------------------------------
# 3. Opérations arithmétiques entre lena et B
# ------------------------------------------------------------
addition = cv2.add(lena, B)
soustraction = cv2.subtract(lena, B)
multiplication = cv2.multiply(lena, B, scale=1/255)  # normalisation pour éviter débordement

# ------------------------------------------------------------
# 4. Opérations logiques entre lena et B
# ------------------------------------------------------------


# ------------------------------------------------------------
# 5. Afficher les résultats
# ------------------------------------------------------------
plt.figure(figsize=(12, 8))

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
plt.title("Addition lena + B")
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(soustraction, cmap='gray')
plt.title("Soustraction lena - B")
plt.axis('off')

plt.subplot(2, 4, 5)
plt.imshow(multiplication, cmap='gray')
plt.title("Multiplication lena * B")
plt.axis('off')



plt.tight_layout()
plt.show()
