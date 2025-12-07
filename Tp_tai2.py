import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# --- 0. FONCTIONS UTILES (Pour les parties 1 et 2) ---

def creer_masque_binaire(largeur, hauteur):
    """
    Crée une image binaire (0 ou 255) avec un rectangle blanc positionné aléatoirement.
    Retourne une image 3-canaux (BGR) pour les opérations avec l'image couleur.
    """
    # Crée un tableau NumPy d'image noire (3 canaux BGR)
    masque = np.zeros((hauteur, largeur, 3), dtype=np.uint8)

    # Définir aléatoirement la taille et la position du rectangle
    w_rect = np.random.randint(50, largeur // 3)
    h_rect = np.random.randint(50, hauteur // 3)

    x1 = np.random.randint(0, largeur - w_rect)
    y1 = np.random.randint(0, hauteur - h_rect)
    x2 = x1 + w_rect
    y2 = y1 + h_rect

    # Dessiner le rectangle blanc (valeur 255) et le remplir
    cv2.rectangle(masque, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=-1)
    return masque

def HISTO(img):
    """
    INSTRUCTION 1 (Partie 2) : Calcule l'histogramme de niveau de gris d'une image 8-bit.
    """
    h, w = img.shape
    hist = np.zeros(256, dtype=int)
    for y in range(h):
        for x in range(w):
            hist[img[y, x]] += 1
    return hist

def TRL(img, C):
    """
    INSTRUCTION 4 (Partie 2) : Applique une Translation Linéaire (changement de luminosité)
    en ajoutant une valeur constante C à tous les pixels.
    """
    # Convertir en int16 pour éviter l'overflow/underflow
    img_trl = img.astype(np.int16) + C
    
    # Clamper les valeurs entre 0 et 255 (Saturation)
    img_trl = np.clip(img_trl, 0, 255)
    
    return img_trl.astype(np.uint8)

def contraste_expansion(img):
    """
    INSTRUCTION 13 (Partie 2) : Implémente la formule pour l'étirement de contraste linéaire.
    I'(i,j) = [255 / (Imax - Imin)] * (I(i,j) - Imin)
    """
    # INSTRUCTION 14 : Identifier le minimum et le maximum des valeurs
    Imin = np.min(img)
    Imax = np.max(img)

    # Éviter la division par zéro
    if Imax == Imin:
        return img.astype(np.uint8)

    # Appliquer la formule d'expansion
    expanded = (255.0 / (Imax - Imin)) * (img.astype(np.float32) - Imin)
    
    return expanded.astype(np.uint8)


# --- 1. CHARGEMENT ET PRÉPARATION DES DONNÉES (Instructions 3 & 4) ---

# Charger Lena en couleur pour la Partie 1 (Opérations arithmétiques/logiques)
lena_color = cv2.imread('lena.jpg', cv2.IMREAD_COLOR) 
# Charger les images en niveaux de gris pour la Partie 2 (Histogramme/Contraste/Seuillage)
lena_gray = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
alex_gray = cv2.imread('alex.png', cv2.IMREAD_GRAYSCALE)

if lena_color is None or lena_gray is None or alex_gray is None:
    print("Erreur: Impossible de charger les images. Assurez-vous que 'lena.jpg' et 'alex.png' sont présents.")
    exit()

hauteur, largeur, _ = lena_color.shape

# INSTRUCTION 5 : Créer l'image binaire B (en 3 canaux pour la Partie 1)
masque_binaire_B = creer_masque_binaire(largeur, hauteur)

# Conversion pour l'affichage Matplotlib
lena_rgb = cv2.cvtColor(lena_color, cv2.COLOR_BGR2RGB)


# ######################################################################
# ### PARTIE 1 : OPÉRATIONS ARITHMÉTIQUES ET LOGIQUES (Instructions 6-8) ###
# ######################################################################

# --- OPÉRATIONS ARITHMÉTIQUES (Instruction 6) ---
img_somme = cv2.add(lena_color, masque_binaire_B)          # Lena + B (Éclaircissement)
img_soustraction = cv2.subtract(lena_color, masque_binaire_B) # Lena - B (Assombrissement)
img_multiplication = cv2.multiply(lena_color, masque_binaire_B, scale=1/255.0) # Lena * B (Masquage)

# Conversion pour l'affichage
img_somme_rgb = cv2.cvtColor(img_somme, cv2.COLOR_BGR2RGB)
img_soustraction_rgb = cv2.cvtColor(img_soustraction, cv2.COLOR_BGR2RGB)
img_multiplication_rgb = cv2.cvtColor(img_multiplication.astype(np.uint8), cv2.COLOR_BGR2RGB)

# --- OPÉRATIONS LOGIQUES (Instruction 7) ---
img_et = cv2.bitwise_and(lena_color, masque_binaire_B)     # Lena ET B
img_ou = cv2.bitwise_or(lena_color, masque_binaire_B)      # Lena OU B
img_xor = cv2.bitwise_xor(lena_color, masque_binaire_B)    # Lena XOR B

# Conversion pour l'affichage
img_et_rgb = cv2.cvtColor(img_et, cv2.COLOR_BGR2RGB)
img_ou_rgb = cv2.cvtColor(img_ou, cv2.COLOR_BGR2RGB)
img_xor_rgb = cv2.cvtColor(img_xor, cv2.COLOR_BGR2RGB)

# --- AFFICHAGE PARTIE 1 (Instruction 8) ---
plt.figure(figsize=(15, 10))
plt.suptitle('PARTIE 1: Opérations Arithmétiques et Logiques', fontsize=16)

# Affichage des 8 images dans une seule figure pour la clarté
plt.subplot(3, 3, 1); plt.imshow(lena_rgb); plt.title('1. Image Originale (Lena)'); plt.axis('off')
plt.subplot(3, 3, 2); plt.imshow(masque_binaire_B); plt.title('2. Masque Binaire (B)'); plt.axis('off')
plt.subplot(3, 3, 3); plt.imshow(img_somme_rgb); plt.title('3. Addition (Lena + B)'); plt.axis('off')
plt.subplot(3, 3, 4); plt.imshow(img_soustraction_rgb); plt.title('4. Soustraction (Lena - B)'); plt.axis('off')
plt.subplot(3, 3, 5); plt.imshow(img_multiplication_rgb); plt.title('5. Multiplication (Lena * B)'); plt.axis('off')
plt.subplot(3, 3, 6); plt.imshow(img_et_rgb); plt.title('6. Logique ET (AND)'); plt.axis('off')
plt.subplot(3, 3, 7); plt.imshow(img_ou_rgb); plt.title('7. Logique OU (OR)'); plt.axis('off')
plt.subplot(3, 3, 8); plt.imshow(img_xor_rgb); plt.title('8. Logique XOR'); plt.axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# #################################################################
# ### PARTIE 2 : MANIPULATION DE L'HISTOGRAMME (Instructions 1-19) ###
# #################################################################

# --- 2.1 HISTOGRAMME DE BASE ET TRANSLATION (Instructions 1-9) ---

# INSTRUCTION 2 & 3 : Calcul et affichage de l'histogramme de Lena
hist_lena = HISTO(lena_gray)

plt.figure(figsize=(10, 4))
plt.suptitle('PARTIE 2.1: Histogramme et Translation Linéaire', fontsize=16)

plt.subplot(1, 2, 1); plt.imshow(lena_gray, cmap='gray'); plt.title("9. Image originale"); plt.axis('off')
plt.subplot(1, 2, 2); plt.plot(hist_lena, color='black'); plt.title("10. Histogramme original"); plt.xlabel("Niveaux de gris"); plt.ylabel("Fréquence")
plt.show()

# INSTRUCTION 5 & 6 & 9 : Application de TRL et affichage
lena_plus = TRL(lena_gray, 50)
lena_minus = TRL(lena_gray, -50)

plt.figure(figsize=(12, 6))
plt.suptitle('Translation Linéaire des Niveaux de Gris (Luminosité)', fontsize=14)

plt.subplot(2, 3, 1); plt.imshow(lena_gray, cmap='gray'); plt.title("11. Originale"); plt.axis('off')
plt.subplot(2, 3, 2); plt.imshow(lena_plus, cmap='gray'); plt.title("12. C = +50 (Plus clair)"); plt.axis('off')
plt.subplot(2, 3, 3); plt.imshow(lena_minus, cmap='gray'); plt.title("13. C = -50 (Plus sombre)"); plt.axis('off')

plt.subplot(2, 3, 4); plt.plot(HISTO(lena_gray), color='black'); plt.title("Histo Original")
plt.subplot(2, 3, 5); plt.plot(HISTO(lena_plus), color='black'); plt.title("Histo +50 (Décalage à droite)")
plt.subplot(2, 3, 6); plt.plot(HISTO(lena_minus), color='black'); plt.title("Histo -50 (Décalage à gauche)")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# --- 2.2 INVERSION (NÉGATIF) (Instructions 10-12) ---

# INSTRUCTION 10 & 11 : Implémenter et appliquer l'inversion (255 - I)
inversion = 255 - lena_gray

# INSTRUCTION 12 : Afficher les images et les histogrammes
plt.figure(figsize=(12, 6))
plt.suptitle('PARTIE 2.2: Inversion de l\'Image (Négatif)', fontsize=16)

plt.subplot(2, 2, 1); plt.imshow(lena_gray, cmap='gray'); plt.title("14. Image Originale"); plt.axis('off')
plt.subplot(2, 2, 2); plt.imshow(inversion, cmap='gray'); plt.title("15. Image Inversée"); plt.axis('off')

plt.subplot(2, 2, 3); plt.plot(HISTO(lena_gray), color='black'); plt.title("Histo Original")
plt.subplot(2, 2, 4); plt.plot(HISTO(inversion), color='black'); plt.title("Histo Inversé (Miroir)")

plt.tight_layout()
plt.show()


# --- 2.3 ÉTIREMENT DE CONTRASTE (Instructions 13-16) ---

# INSTRUCTION 15 : Appliquer l'expansion de l'histogramme à alex.png
alex_expanded = contraste_expansion(alex_gray)

# INSTRUCTION 16 : Afficher les résultats avec les histogrammes respectifs
plt.figure(figsize=(12, 6))
plt.suptitle('PARTIE 2.3: Étirement de Contraste (Expansion d\'Histogramme)', fontsize=16)
print(f"Valeurs min et max de l'image 'Alex' après expansion : {np.min(alex_expanded)}, {np.max(alex_expanded)}") # Vérification

plt.subplot(2, 2, 1); plt.imshow(alex_gray, cmap='gray'); plt.title("16. Alex Originale (Contraste faible)"); plt.axis('off')
plt.subplot(2, 2, 2); plt.imshow(alex_expanded, cmap='gray'); plt.title("17. Après Étirement du Contraste"); plt.axis('off')

plt.subplot(2, 2, 3); plt.plot(HISTO(alex_gray), color='black'); plt.title("Histo Original (Condensé)")
plt.subplot(2, 2, 4); plt.plot(HISTO(alex_expanded), color='black'); plt.title("Histo Après Expansion (Étiré de 0 à 255)")

plt.tight_layout()
plt.show()


# --- 2.4 SEUILLAGE (Instructions 17-19) ---

T_manuel = 120 # INSTRUCTION 17 : Seuil choisi manuellement

# Seuillage simple (manuel)
_, lena_thresh_manuel = cv2.threshold(lena_gray, T_manuel, 255, cv2.THRESH_BINARY)
_, alex_thresh_manuel = cv2.threshold(alex_gray, T_manuel, 255, cv2.THRESH_BINARY)

# INSTRUCTION 18 : Seuillage automatique (Méthode d'Otsu)
T_otsu_lena, lena_otsu = cv2.threshold(lena_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
T_otsu_alex, alex_otsu = cv2.threshold(alex_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print("\n--- Seuillage Automatique (Otsu) ---")
print(f"Seuil Otsu calculé pour Lena : {int(T_otsu_lena)}")
print(f"Seuil Otsu calculé pour Alex : {int(T_otsu_alex)}")

# Affichage de la comparaison (Instructions 17, 18 et 19)
plt.figure(figsize=(15, 6))
plt.suptitle('PARTIE 2.4: Comparaison des Seuillages (Manuel vs Otsu)', fontsize=16)

# Résultats Lena
plt.subplot(2, 3, 1); plt.imshow(lena_gray, cmap='gray'); plt.title("18. Lena Originale"); plt.axis('off')
plt.subplot(2, 3, 2); plt.imshow(lena_thresh_manuel, cmap='gray'); plt.title(f"19. Manuel (T={T_manuel})"); plt.axis('off')
plt.subplot(2, 3, 3); plt.imshow(lena_otsu, cmap='gray'); plt.title(f"20. Otsu (T={int(T_otsu_lena)})"); plt.axis('off')

# Résultats Alex
plt.subplot(2, 3, 4); plt.imshow(alex_gray, cmap='gray'); plt.title("21. Alex Originale"); plt.axis('off')
plt.subplot(2, 3, 5); plt.imshow(alex_thresh_manuel, cmap='gray'); plt.title(f"22. Manuel (T={T_manuel})"); plt.axis('off')
plt.subplot(2, 3, 6); plt.imshow(alex_otsu, cmap='gray'); plt.title(f"23. Otsu (T={int(T_otsu_alex)})"); plt.axis('off')

plt.tight_layout()
plt.show()

# Fermeture des fenêtres OpenCV (Bien que non utilisées ici, on les garde pour la complétude)
cv2.waitKey(0)
cv2.destroyAllWindows()
