import cv2 
import matplotlib.pyplot as plt
import numpy as np

img_path = "lena.jpg"

image_color=cv2.imread(img_path,cv2.IMREAD_COLOR)
img_rgb=cv2.cvtColor(image_color,cv2.COLOR_BGR2RGB)
img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

Img_echantillon = cv2.resize(img_gray, (50, 50), interpolation=cv2.INTER_NEAREST)




plt.figure(figsize=(5,5))
plt.title("Lena sous-échantillonnée")
plt.imshow(Img_echantillon, cmap='gray')
plt.axis('off')
plt.show()



def _imgquantif(img,k):
  step=256 //k
  quantized = (img // step) * step + step // 2
  quantized = np.clip(quantized, 0, 255)  
  return quantized.astype(np.uint8)



quant_img = _imgquantif(img_gray, 4)
plt.imshow(quant_img, cmap='gray')

plt.suptitle("Quantification de l'image ")
plt.show()



