import numpy as np
import cv2
class CountBox:

    def binarize(self, image, threshold):
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        return binary.astype(bool)

    def pad(self, image):
        height, width = image.shape
        h_pad = 2**np.ceil(np.log2(height)).astype(int)
        w_pad = 2**np.ceil(np.log2(width)).astype(int)
        padded = np.zeros((h_pad, w_pad), dtype=bool)
        padded[:height, :width] = image
        return padded

    # Define a function to count the number of boxes of a given size that contain the object
    def box_count(self, image, size):
        height, width = image.shape
        count = 0
        for i in range(0, height, size):
            for j in range(0, width, size):
                window = image[i:i+size, j:j+size]
                if np.any(window):
                    count += 1
        return count

    # Define a function to calculate the Hausdorff fractal dimension of an image
    def hausdorff(self, image, threshold=128):
        image = self.binarize(image, threshold)
        image = self.pad(image)
        size = image.shape[0]
        sizes = []
        counts = []
        while size > 0:
            sizes.append(size)
            count = self.box_count(image, size)
            counts.append(count)
            size //= 2
        sizes = np.array(sizes)
        counts = np.array(counts)
        slope, intercept = np.polyfit(np.log(1.0 / sizes), np.log(counts), 1)
        return slope, intercept, sizes