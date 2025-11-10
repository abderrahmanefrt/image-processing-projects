import cv2
import numpy as np
import matplotlib.pyplot as plt

# Fonction de convolution avec padding (de l'exercice précédent)
def convolution_avec_padding(img, kernel):
    """
    Applique une convolution avec padding sur une image
    """
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
            # Extraire la région
            region = padded[i:i+k, j:j+k]
            # Calculer le produit élément par élément et sommer
            output[i, j] = np.sum(region * kernel)
    
    return output

# Question 2: Fonction pour créer un filtre gaussien
def filtre_Gaussien(sigma, taille):
    """
    Crée un noyau de convolution gaussien
    
    Paramètres:
    - sigma: écart-type de la distribution gaussienne
    - taille: taille du noyau (doit être impair)
    
    Retourne:
    - kernel: noyau gaussien normalisé
    """
    # Créer une grille de coordonnées
    ax = np.arange(-taille // 2 + 1, taille // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    
    # Calculer le noyau gaussien selon la formule
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    
    # Normaliser le noyau pour que la somme soit égale à 1
    kernel = kernel / np.sum(kernel)
    
    return kernel

# Charger l'image
img = cv2.imread('lena_noise.jpg', cv2.IMREAD_GRAYSCALE)

# Question 3: Afficher les valeurs du filtre gaussien pour K=3 et sigma=1
print("Question 3: Filtre Gaussien avec K=3 et σ=1")

kernel_3x3 = filtre_Gaussien(sigma=1, taille=3)
print("Valeurs du noyau gaussien 3x3:")
print(kernel_3x3)
print(f"\nSomme des coefficients: {np.sum(kernel_3x3):.6f}")


# Question 4 et 5: Appliquer le filtre gaussien avec K=3 et sigma=1
print("\nApplication du filtre gaussien...")
img_filtered_3x3 = convolution_avec_padding(img, kernel_3x3)

# Afficher l'image filtrée
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

# Question 6 et 7: Tester différentes tailles de noyaux et différents sigma
print("\n Test avec différents paramètres...")

tailles = [3, 5, 7]
sigmas = [0.5, 1, 2]

# Créer une figure pour afficher tous les résultats
fig, axes = plt.subplots(len(sigmas), len(tailles), figsize=(15, 12))
fig.suptitle('Filtre Gaussien avec différents paramètres', fontsize=16)

for i, sigma in enumerate(sigmas):
    for j, taille in enumerate(tailles):
        # Créer le filtre gaussien
        kernel = filtre_Gaussien(sigma, taille)
        
        # Appliquer la convolution
        img_filtered = convolution_avec_padding(img, kernel)
        
        # Afficher
        axes[i, j].imshow(img_filtered, cmap='gray')
        axes[i, j].set_title(f'K={taille}, σ={sigma}')
        axes[i, j].axis('off')
        

plt.tight_layout()
plt.show()

