from skimage import io
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import rescale, resize
from skimage.filters import threshold_otsu
from skimage.feature import match_template
import numpy as np

# Load the images
coins_image = io.imread('/home/aditya/Documents/pythonLabZeroToFour/labFour/coins.jpg')
astronaut_image = io.imread('/home/aditya/Documents/pythonLabZeroToFour/labFour/astronaut.jpg')

# Display the images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(coins_image)
plt.title('Coins Image')

plt.subplot(1, 2, 2)
plt.imshow(astronaut_image)
plt.title('Astronaut Image')

plt.show()

# Check image properties
print("Coins image shape:", coins_image.shape)
print("Astronaut image shape:", astronaut_image.shape)
print("Coins image dtype:", coins_image.dtype)
print("Astronaut image dtype:", astronaut_image.dtype)

# Accessing a pixel's intensity level
print("Intensity level at [1, 100, 1] in Coins image:", coins_image[1, 100, 1])
print("Intensity level at [1, 100, 1] in Astronaut image:", astronaut_image[1, 100, 1])

# Convert the images to grayscale
coins_gray = rgb2gray(coins_image)
astronaut_gray = rgb2gray(astronaut_image)

# Display the grayscale images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(coins_gray, cmap='gray')
plt.title('Coins Grayscale Image')

plt.subplot(1, 2, 2)
plt.imshow(astronaut_gray, cmap='gray')
plt.title('Astronaut Grayscale Image')

plt.show()

# Check the dtype and size before and after conversion
print("Coins grayscale image shape:", coins_gray.shape)
print("Astronaut grayscale image shape:", astronaut_gray.shape)
print("Coins grayscale image dtype:", coins_gray.dtype)
print("Astronaut grayscale image dtype:", astronaut_gray.dtype)

# Resize the astronaut image
astronaut_resized = resize(astronaut_image, (100, 100))

# Rescale the astronaut image
astronaut_rescaled_075 = rescale(astronaut_image, 0.75)
astronaut_rescaled_05 = rescale(astronaut_image, 0.5)
astronaut_rescaled_025 = rescale(astronaut_image, 0.25)

# Display the resized and rescaled images
plt.figure(figsize=(10, 5))
plt.subplot(1, 4, 1)
plt.imshow(astronaut_resized)
plt.title('Resized Image (100x100)')

plt.subplot(1, 4, 2)
plt.imshow(astronaut_rescaled_075)
plt.title('Rescaled 0.75')

plt.subplot(1, 4, 3)
plt.imshow(astronaut_rescaled_05)
plt.title('Rescaled 0.5')

plt.subplot(1, 4, 4)
plt.imshow(astronaut_rescaled_025)
plt.title('Rescaled 0.25')

plt.show()

# Use the grayscale converted coins image and apply Otsu's threshold
t = threshold_otsu(coins_gray)

# Apply threshold to the coins image
binary_coins = coins_gray > t

# Display the binary image
plt.figure(figsize=(5, 5))
plt.imshow(binary_coins, cmap='gray')
plt.title('Coins Image after Thresholding')
plt.show()

# Load the template (a portion of the coins image)
template = coins_gray[30:100, 30:100]  # A section of the image

# Perform template matching
result = match_template(coins_gray, template)

# Find the location with the highest similarity
ij = np.unravel_index(np.argmax(result), result.shape)
x, y = ij[::-1]

# Display the result
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(template, cmap='gray')
ax[0].set_title('Template')

ax[1].imshow(coins_gray, cmap='gray')
ax[1].set_title('Detected Template Location')
rect = plt.Rectangle((x, y), template.shape[1], template.shape[0], edgecolor='r', facecolor='none')
ax[1].add_patch(rect)
plt.show()

# Template matching using a sliding window
def template_matching(image, template):
    best_score = -1
    best_position = (0, 0)

    # Sliding window over the image
    for i in range(image.shape[0] - template.shape[0]):
        for j in range(image.shape[1] - template.shape[1]):
            region = image[i:i + template.shape[0], j:j + template.shape[1]]
            score = np.sum(region * template)  # Simple cross-correlation

            if score > best_score:
                best_score = score
                best_position = (i, j)

    return best_position

# Load the grayscale image and template for manual matching
template = coins_gray[30:100, 30:100]

# Perform template matching
best_pos = template_matching(coins_gray, template)
print(f'Best match found at position: {best_pos}')
