import cv2
import numpy as np
import matplotlib.pyplot as plt

def convolution_avec_padding(img, kernel):
    
    # Inverser le kernel (convolution)
    kernel = np.flip(kernel)
    
    # Calculer le padding nécessaire
    k = kernel.shape[0]
    p = k // 2
    
    # Ajouter le padding
    padded = np.pad(img, ((p, p), (p, p)), mode='constant', constant_values=0)
    
    # Dimensions de sortie
    M, N = img.shape
    output = np.zeros((M, N))
    
    # Appliquer la convolution
    for i in range(M):
        for j in range(N):
            region = padded[i:i+k, j:j+k]
            output[i, j] = np.sum(region * kernel)
    
    return output

def filtre_Gaussien(sigma, taille):

    ax = np.arange(-taille // 2 + 1, taille // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    
    kernel = kernel / np.sum(kernel)
    
    return kernel

img = cv2.imread('lena_noise.jpg', cv2.IMREAD_GRAYSCALE)

print("Question 3: Filtre Gaussien avec K=3 et σ=1")

kernel_3x3 = filtre_Gaussien(sigma=1, taille=3)
print("Valeurs du noyau gaussien 3x3:")
print(kernel_3x3)
print(f"\nSomme des coefficients: {np.sum(kernel_3x3):.6f}")


print("\nApplication du filtre gaussien...")
img_filtered_3x3 = convolution_avec_padding(img, kernel_3x3)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Image originale (bruitée)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_filtered_3x3, cmap='gray')
plt.title('Image filtrée (K=3, σ=1)')
plt.axis('off')
plt.tight_layout()
plt.show()

print("\n Test avec différents paramètres...")

tailles = [3, 5, 7]
sigmas = [0.5, 1, 2]

fig, axes = plt.subplots(len(sigmas), len(tailles), figsize=(15, 12))
fig.suptitle('Filtre Gaussien avec différents paramètres', fontsize=16)

for i, sigma in enumerate(sigmas):
    for j, taille in enumerate(tailles):
        kernel = filtre_Gaussien(sigma, taille)
        
        img_filtered = convolution_avec_padding(img, kernel)
        
        axes[i, j].imshow(img_filtered, cmap='gray')
        axes[i, j].set_title(f'K={taille}, σ={sigma}')
        axes[i, j].axis('off')
        

plt.tight_layout()
plt.show()

