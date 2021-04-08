from skimage.feature import hog
from skimage import data, exposure
import matplotlib.pyplot as plt

image = data.astronaut()

fd, hog_img = hog(image, orientations=8, pixels_per_cell=(16,16), 
              cells_per_block=(1,1), visualize=True, multichannel=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4), sharex=True, sharey=True)

ax1.axis("off")
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title("input_title")

rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 10))

ax2.axis("off")
ax2.imshow(rescaled, cmap=plt.cm.gray)
ax2.set_title("histogram of Oriented Gradients")
plt.show()

 

