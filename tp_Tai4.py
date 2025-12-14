import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 0. PRÉPARATION ---

# Charger l'image I (catty.jpg) en niveaux de gris pour l'analyse des gradients
img_original = cv2.imread('catty.jpg', cv2.IMREAD_GRAYSCALE)

if img_original is None:
    print("FATAL ERROR: Impossible de charger 'catty.jpg'. Vérifiez le chemin du fichier.")
    exit()

# Conversion de l'image en float32 pour les calculs de convolution (prévient l'overflow)
img_float = img_original.astype(np.float32)

# Définition pour l'affichage
fig_count = 1

# ######################################################################
# ### PARTIE 1 : GRADIENTS (DÉTECTEUR DE SOBEL) (Instructions 5-8) #####
# ######################################################################

# 5. Définition des filtres de Sobel (Instructions 5)
Gx = np.array([[-1, 0, 1], 
               [-2, 0, 2], 
               [-1, 0, 1]], dtype=np.float32) # Détection des bords VERTICAUX [cite: 54]

Gy = np.array([[-1, -2, -1], 
               [ 0,  0,  0], 
               [ 1,  2,  1]], dtype=np.float32) # Détection des bords HORIZONTAUX [cite: 54]

# 6. Appliquer les filtres Gx et Gy sur l'image I
# cv2.filter2D est la fonction de convolution générale.
# ddepth=-1 signifie que la sortie aura la même profondeur que l'entrée (float32 ici)
grad_x = cv2.filter2D(img_float, -1, Gx)
grad_y = cv2.filter2D(img_float, -1, Gy)

# 7. Calculez la magnitude du gradient G (G = sqrt(Gx^2 + Gy^2)) [cite: 62]
G_magnitude = np.sqrt(grad_x**2 + grad_y**2)

# 8. Normalisez et affichez l'image résultat

# Normalisation de la magnitude du gradient (échelle de 0 à 255)
# Nécessaire car les valeurs de convolution peuvent dépasser 255 ou être négatives.
G_normalized = cv2.normalize(G_magnitude, None, 0, 255, cv2.NORM_MINMAX)
G_uint8 = np.uint8(G_normalized)


plt.figure(figsize=(15, 5))
plt.suptitle('PARTIE 1: Détection de Contours par Gradients (Sobel)', fontsize=16)

plt.subplot(1, 3, 1); plt.imshow(np.uint8(cv2.normalize(grad_x, None, 0, 255, cv2.NORM_MINMAX)), cmap='gray'); plt.title('1. Gradient Gx (Verticaux)'); plt.axis('off')
plt.subplot(1, 3, 2); plt.imshow(np.uint8(cv2.normalize(grad_y, None, 0, 255, cv2.NORM_MINMAX)), cmap='gray'); plt.title('2. Gradient Gy (Horizontaux)'); plt.axis('off')
plt.subplot(1, 3, 3); plt.imshow(G_uint8, cmap='gray'); plt.title('3. Magnitude du Gradient G'); plt.axis('off')
plt.show()

# ######################################################################
# ### PARTIE 2 : LE FILTRE LAPLACIEN (Instructions 1-4) ################
# ######################################################################

# Noyau Laplacien (Instruction 2. Soit L le noyau Laplacien) [cite: 68]
L_noyau = np.array([[0, -1, 0], 
                    [-1, 4, -1], 
                    [0, -1, 0]], dtype=np.float32)

# 1. Appliquer le masque Laplacien sur l'image I
img_laplacian_scratch = cv2.filter2D(img_float, -1, L_noyau)

# 2. Visualiser l'image Laplacienne (normaliser le résultat)
laplacian_norm = cv2.normalize(img_laplacian_scratch, None, 0, 255, cv2.NORM_MINMAX)
laplacian_uint8 = np.uint8(laplacian_norm)

# 3. Appliquer un filtre gaussien (prélissage) puis le Laplacien (LoG - Laplacian of Gaussian)
# Prélissage Gaussien (Réduit le bruit, très important pour le Laplacien)
img_gauss = cv2.GaussianBlur(img_float, (5, 5), 1.5)
img_log = cv2.filter2D(img_gauss, -1, L_noyau)

log_norm = cv2.normalize(img_log, None, 0, 255, cv2.NORM_MINMAX)
log_uint8 = np.uint8(log_norm)

# 4. Appliquer le Laplacien prédéfini dans OpenCV
# ddepth=cv2.CV_16S est souvent utilisé car le Laplacien produit des valeurs négatives.
laplacian_cv = cv2.Laplacian(img_original, cv2.CV_16S, ksize=3)
laplacian_cv_norm = cv2.normalize(laplacian_cv, None, 0, 255, cv2.NORM_MINMAX)
laplacian_cv_uint8 = np.uint8(laplacian_cv_norm)


plt.figure(figsize=(15, 5))
plt.suptitle('PARTIE 2: Filtre Laplacien (Dérivée Seconde)', fontsize=16)

plt.subplot(1, 4, 1); plt.imshow(laplacian_uint8, cmap='gray'); plt.title('4. Laplacien Brut (Scratch)'); plt.axis('off')
plt.subplot(1, 4, 2); plt.imshow(img_gauss, cmap='gray'); plt.title('5. Image Gaussiensée'); plt.axis('off')
plt.subplot(1, 4, 3); plt.imshow(log_uint8, cmap='gray'); plt.title('6. LoG (Laplacien après Gaussien)'); plt.axis('off')
plt.subplot(1, 4, 4); plt.imshow(laplacian_cv_uint8, cmap='gray'); plt.title('7. Laplacien avec cv2.Laplacian()'); plt.axis('off')
plt.show()


# ######################################################################
# ### PARTIE 3 : DÉTECTEUR DE CANNY (Instructions 2-4) ##################
# ######################################################################

# 1. Fonction OpenCV pour Canny : cv2.Canny() [cite: 82]

# 2. Appliquer le détecteur Canny avec thresholds donnés [cite: 84]
T1_defaut = 100
T2_defaut = 50
edges_defaut = cv2.Canny(img_original, T1_defaut, T2_defaut)

# 4. Appliquer le détecteur de Canny avec différents couples de seuils [cite: 86]
T1_faible = 20
T2_faible = 10
edges_faible = cv2.Canny(img_original, T1_faible, T2_faible) # Beaucoup de bruit

T1_fort = 200
T2_fort = 150
edges_fort = cv2.Canny(img_original, T1_fort, T2_fort) # Moins de contours


# --- AFFICHAGE PARTIE 3 ---
plt.figure(figsize=(15, 5))
plt.suptitle('PARTIE 3: Détecteur de Canny (Seuillage par Hystérésis)', fontsize=16)

plt.subplot(1, 4, 1); plt.imshow(img_original, cmap='gray'); plt.title('8. Originale'); plt.axis('off')
plt.subplot(1, 4, 2); plt.imshow(edges_defaut, cmap='gray'); plt.title(f'9. Canny (T1={T1_defaut}, T2={T2_defaut})'); plt.axis('off')
plt.subplot(1, 4, 3); plt.imshow(edges_faible, cmap='gray'); plt.title(f'10. Canny Faible (Bruit)'); plt.axis('off')
plt.subplot(1, 4, 4); plt.imshow(edges_fort, cmap='gray'); plt.title(f'11. Canny Fort (Contours manquants)'); plt.axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
