import cv2 
import matplotlib.pyplot as plt
import numpy as np

img_path = "lena.jpg"

image_color=cv2.imread(img_path,cv2.IMREAD_COLOR)
img_rgb=cv2.cvtColor(image_color,cv2.COLOR_BGR2RGB)

b,g,r=cv2.split(img_rgb)

plt.figure(figsize=(10,4))

plt.subplot(1,3,1)

plt.title("Canal Rouge")

plt.imshow(r, cmap="gray")

plt.subplot(1,3,2)

plt.title("Canal vert")

plt.imshow(g, cmap="gray")

plt.subplot(1,3,3)

plt.imshow(b, cmap="gray")

plt.title("Canal bleu")

plt.show()


