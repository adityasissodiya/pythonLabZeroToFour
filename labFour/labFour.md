### Lab 4: Getting Started with Image Processing Using Scikit-Image

In this lab, you will use the **Scikit-Image** library to perform various image processing tasks. You’ll start by loading and visualizing images, and then move on to operations like color space conversion, resizing, thresholding, and template matching.

---

### **Task 0: Software Setup**
Before starting, you need to install the Scikit-Image library. You can do so with the following command:

```bash
pip install scikit-image
```

---

### **Task 1: Load and Visualize an Image**

1. **Loading and Visualizing Images**:
   Start by creating a new Python file and load the provided images (`coins.jpg` and `astronaut.jpg`) using `scikit-image`.

#### Code for Task 1: Loading and Visualizing Images

```python
from skimage import io
import matplotlib.pyplot as plt

# Load the images
coins_image = io.imread('coins.jpg')
astronaut_image = io.imread('astronaut.jpg')

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
```

#### Explanation:
- **`.shape`**: Shows the dimensions of the image (height, width, channels).
- **`.dtype`**: Tells the data type of the pixel values (e.g., `uint8` or `float64`).
- **Pixel Access**: Accessing a specific pixel to get its intensity value.

---

### **Task 2: Color Space Conversion**

In this task, you’ll convert color images to grayscale using `scikit-image`.

#### Code for Task 2: Converting Images to Grayscale

```python
from skimage.color import rgb2gray

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
```

#### Explanation:
- **Before Conversion**: Both images are in RGB format with 3 channels.
- **After Conversion**: Grayscale images will have only one channel, reducing the size. The dtype may change, typically from `uint8` (0-255) to `float64` (0-1).

---

### **Task 3: Image Rescale and Resize**

Next, you will modify the size and scale of images using the `resize` and `rescale` functions.

#### Code for Task 3: Resizing and Rescaling Images

```python
from skimage.transform import rescale, resize

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
```

#### Explanation:
- **`resize`**: Changes the image dimensions to the specified size.
- **`rescale`**: Scales the image by a factor, e.g., 0.75 scales the image to 75% of its original size.

---

### **Task 4: Image Thresholding**

Thresholding is used for image segmentation. In this task, you will convert the grayscale version of the "coins" image to binary using thresholding.

#### Code for Task 4: Thresholding

```python
from skimage.filters import threshold_otsu
import numpy as np

# Use the grayscale converted coins image
t = threshold_otsu(coins_gray)

# Apply threshold
binary_coins = coins_gray > t

# Display the binary image
plt.figure(figsize=(5, 5))
plt.imshow(binary_coins, cmap='gray')
plt.title('Coins Image after Thresholding')
plt.show()
```

#### Explanation:
- **Threshold Otsu**: Automatically finds the optimal threshold value.
- **Binary Image**: All pixel values above the threshold become `True` (white), and below become `False` (black).

---

### **Task 5: Template Matching**

Template matching is a technique to find and locate a template image in a larger image. Scikit-image provides a simple way to perform this operation.

#### Code for Task 5: Template Matching

```python
from skimage.feature import match_template

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
```

#### Explanation:
- **Template**: A section of the original image that you want to find within the larger image.
- **`match_template`**: This function finds the best match of the template within the original image.

---

### **Task 6: Own Implementation of Template Matching**

In this task, you will implement your own version of template matching using the concept of sliding window and cross-correlation. Here's a brief outline of how you might approach it:

1. **Sliding Window**: Slide the template over the larger image.
2. **Cross-correlation**: At each position, calculate the similarity between the template and the corresponding region of the image.
3. **Best Match**: The position with the highest similarity is the best match.

Here's an example of how to approach it:

```python
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

# Load the grayscale image and template
template = coins_gray[30:100, 30:100]

# Perform template matching
best_pos = template_matching(coins_gray, template)
print(f'Best match found at position: {best_pos}')
```

This is a basic implementation of template matching without using `scikit-image`. The score is calculated as the sum of element-wise multiplication (cross-correlation).

---

### Summary

In this lab, you:
- Installed and used the **scikit-image** library.
- Loaded and visualized images.
- Converted images to grayscale.
- Resized and rescaled images.
- Applied thresholding for segmentation.
- Used template matching for locating objects in an image.
- Implemented your own template matching algorithm.

![alt text](https://github.com/adityasissodiya/pythonLabZeroToFour/blob/main/labFour/labFour_1.png)
![alt text](https://github.com/adityasissodiya/pythonLabZeroToFour/blob/main/labFour/labFour_2.png)
![alt text](https://github.com/adityasissodiya/pythonLabZeroToFour/blob/main/labFour/labFour_3.png)
![alt text](https://github.com/adityasissodiya/pythonLabZeroToFour/blob/main/labFour/labFour_4.png)
![alt text](https://github.com/adityasissodiya/pythonLabZeroToFour/blob/main/labFour/labFour_5.png)