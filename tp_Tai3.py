import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 0. FONCTIONS UTILES (RÉUTILISATION ET CRÉATION DE NOYAUX) ---

def filtre_Gaussien(K, sigma):
    """
    PARTIE 3 - INSTRUCTION 2 : Crée un noyau de convolution Gaussien KxK.
    Le noyau est normalisé (somme = 1) pour ne pas changer la luminosité globale.
    """
    if K % 2 == 0:
        raise ValueError("La taille du noyau K doit être impaire.")

    # Calcul de la distance au centre (x et y)
    centre = K // 2
    x = np.arange(-centre, centre + 1)
    y = np.arange(-centre, centre + 1)
    X, Y = np.meshgrid(x, y)

    # Formule de la distribution Gaussienne 2D :
    terme_exp = (X**2 + Y**2) / (2 * sigma**2)
    terme_const = 1 / (2 * np.pi * sigma**2)
    noyau = terme_const * np.exp(-terme_exp)
    
    # Normalisation : la somme du noyau doit être égale à 1
    return noyau / np.sum(noyau)


def convolution_avec_padding(img, kernel):
    """
    PARTIE 2 - INSTRUCTION 5 : Implémente la convolution 2D 'from scratch' (à partir de zéro)
    avec un padding de zéros pour que l'image de sortie garde la même taille que l'image d'entrée.
    """
    img = img.astype(np.float32) # Conversion pour des calculs précis
    h, w = img.shape
    kh, kw = kernel.shape
    
    # Calcul du padding nécessaire P = (K - 1) / 2
    p = (kh - 1) // 2
    
    # Ajout du padding de zéros autour de l'image (Instruction 4)
    padded_img = np.pad(img, ((p, p), (p, p)), mode='constant', constant_values=0)
    
    # Inversion du noyau (étape nécessaire pour une vraie CONVOLUTION)
    kernel_flipped = np.flip(kernel, axis=(0, 1)) 

    # Initialisation de l'image de sortie
    output = np.zeros((h, w), dtype=float) 

    # Double boucle pour parcourir chaque pixel de l'image originale
    for i in range(h):
        for j in range(w):
            # Extraction de la région de l'image (de la taille du noyau KxK)
            region = padded_img[i:i+kh, j:j+kw]
            
            # Opération de convolution : Produit élément par élément (multiplication) et sommation
            output[i, j] = np.sum(region * kernel_flipped)
            
    return output


# --- 1. PRÉPARATION ET CHARGEMENT (Instructions 4 & 5) ---

img_bruit = cv2.imread('lena_noise.jpg', cv2.IMREAD_GRAYSCALE)

if img_bruit is None:
    print("FATAL ERROR: Impossible de charger 'lena_noise.jpg'. Placez le fichier dans le même répertoire.")
    exit()

kernel_sizes = [3, 5, 7, 11] # Tailles de noyaux à tester (Instruction 8)


# ######################################################################
# ### PARTIE 1 : CONVOLUTION AVEC OPENCV (Instructions 6-9) ############
# ######################################################################

results_part1 = {}

for K in kernel_sizes:
    # FILTRE MOYEN (Instruction 6 & 7) : utilise cv2.blur()
    img_mean = cv2.blur(img_bruit, (K, K))
    results_part1[f'Moyen K={K}'] = img_mean
    
    # FILTRE MÉDIAN (Instruction 6 & 7) : utilise cv2.medianBlur()
    # Le filtre médian est efficace contre le bruit "sel et poivre" et préserve mieux les bords.
    if K % 2 != 0:
        img_median = cv2.medianBlur(img_bruit, K)
        results_part1[f'Médian K={K}'] = img_median

# NOYAU QUELCONQUE (Instruction 9) : Détection de bords (Sobel)
kernel_quelconque = np.array([[-1, -2, -1],
                              [ 0,  0,  0],
                              [ 1,  2,  1]], dtype=np.float32)

# cv2.filter2D() est la fonction générale de convolution d'OpenCV
img_quelconque = cv2.filter2D(img_bruit, -1, kernel_quelconque)
results_part1['Noyau Quelconque (Bords)'] = img_quelconque

# --- AFFICHAGE PARTIE 1 ---
plt.figure(figsize=(18, 10))
plt.suptitle('PARTIE 1: Convolution avec OpenCV (Filtres et Tailles de Noyaux)', fontsize=16)

plt.subplot(3, 4, 1); plt.imshow(img_bruit, cmap='gray'); plt.title('Originale Bruitée'); plt.axis('off')

i = 2
for title, img_result in results_part1.items():
    plt.subplot(3, 4, i); plt.imshow(img_result, cmap='gray'); plt.title(title); plt.axis('off')
    i += 1

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# ######################################################################
# ### PARTIE 2 : CONVOLUTION 'FROM SCRATCH' (Instructions 3 & 6) #######
# ######################################################################

# Noyau défini pour l'exercice
kernel_sobel = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=np.float32)

# Application de la fonction corrigée (avec padding)
# Cette fonction est la version 'fait maison' de cv2.filter2D
result_avec_padding = convolution_avec_padding(img_bruit, kernel_sobel)

# Normalisation pour l'affichage (important pour les noyaux qui produisent des valeurs hors 0-255)
result_norm = cv2.normalize(result_avec_padding, None, 0, 255, cv2.NORM_MINMAX)
result_uint8 = np.uint8(result_norm)

# --- AFFICHAGE PARTIE 2 ---
plt.figure(figsize=(10, 5))
plt.suptitle('PARTIE 2: Convolution Implémentée avec Padding', fontsize=16)

plt.subplot(1, 2, 1); plt.imshow(img_bruit, cmap='gray'); plt.title('Originale Bruitée'); plt.axis('off')
plt.subplot(1, 2, 2); plt.imshow(result_uint8, cmap='gray'); plt.title('Convolution "from scratch" (K=3, avec Padding)'); plt.axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# ######################################################################
# ### PARTIE 3 : FILTRE GAUSSIEN (Instructions 3-7) ####################
# ######################################################################

# --- DÉMONSTRATION NOYAU GAUSSIEN (Instruction 3) ---
K_test = 3
sigma_test = 1
noyau_gaussien_3x3 = filtre_Gaussien(K_test, sigma_test)

print(f"\nPARTIE 3 - 3. Valeurs du Filtre Gaussien (K={K_test}, σ={sigma_test}):")
print(noyau_gaussien_3x3.round(5))


# --- APPLICATION AVEC CONVOLUTION 'FROM SCRATCH' (Instruction 4 & 5) ---

# Application du noyau Gaussien K=3, σ=1 via la fonction de convolution faite maison
img_gauss_scratch = convolution_avec_padding(img_bruit, noyau_gaussien_3x3)
img_gauss_scratch_norm = cv2.normalize(img_gauss_scratch, None, 0, 255, cv2.NORM_MINMAX)
img_gauss_scratch_uint8 = np.uint8(img_gauss_scratch_norm)


# --- COMPARAISON DES PARAMÈTRES (Instructions 6 & 7) ---

K_values_comp = [5, 7]
Sigma_values_comp = [0.5, 1, 2]

plt.figure(figsize=(18, 9))
plt.suptitle('PARTIE 3: Effets de la Variation de K et Sigma sur le Flou Gaussien', fontsize=16)

# Affichage de l'originale et du scratch
plt.subplot(2, 4, 1); plt.imshow(img_bruit, cmap='gray'); plt.title('Originale Bruitée'); plt.axis('off')
plt.subplot(2, 4, 2); plt.imshow(img_gauss_scratch_uint8, cmap='gray'); plt.title(f'Gaussien Scratch (K={K_test}, σ={sigma_test})'); plt.axis('off')

plot_index = 3
for K in K_values_comp:
    for sigma in Sigma_values_comp:
        # cv2.GaussianBlur (Instruction 1 & 6) est utilisé pour la comparaison
        img_gauss_cv = cv2.GaussianBlur(img_bruit, (K, K), sigma)
        
        plt.subplot(2, 4, plot_index)
        plt.imshow(img_gauss_cv, cmap='gray')
        plt.title(f'K={K} | σ={sigma} (Plus grand K ou σ = Plus flou)')
        plt.axis('off')
        plot_index += 1

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
