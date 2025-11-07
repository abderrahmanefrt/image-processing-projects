import cv2
import matplotlib.pyplot as plt
import numpy as np



img = cv2.imread("lena_noise.jpg", cv2.IMREAD_GRAYSCALE)

kernel = np.array([[1, 0, -1],
                   [0, 1, 0],
                   [1, 0, -1]], dtype=np.float64)

def convolution(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape

    kernel = np.flipud(np.fliplr(kernel))

    new_h = h - kh + 1
    new_w = w - kw + 1

    output = np.zeros((new_h, new_w), dtype=np.float64)


    for i in range(new_h):
        for j in range(new_w):
            region = img[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)

    return output


result = convolution(img, kernel)

result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
result = np.uint8(result)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("originale")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(result, cmap="gray")
plt.title("convolution")
plt.axis("off")
plt.tight_layout()
plt.show()


'''
Ce code applique une convolution manuelle sur une image en niveaux de gris (image.jpg) avec un noyau défini manuellement

 Pas d’inversion du noyau
→ Le code effectue une corrélation (region * kernel) au lieu d’une convolution.
→ Il faut retourner le noyau sur les deux axes (np.flipud(np.fliplr(kernel))).

 La somme des produits est manquante
→ La ligne output[i, j] = region * kernel donne une matrice.
→ Il faut faire la somme (np.sum(region * kernel)).

 Types de données
→ Le type float sans précision peut poser problème. Il vaut mieux utiliser np.float32.

 Dimensions de sortie incorrectes
→ Le code calcule new_h = h - kh + 1 et new_w = w - kw + 1, donc l’image filtrée est plus petite.
→ Pour garder la même taille, il faut ajouter du padding (optionnel selon l’énoncé).'''


