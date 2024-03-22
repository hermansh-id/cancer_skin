import cv2

class Preprocess:


    def process(self, 
                image, 
                ksize=5,
                canny_low=100,
                canny_high=200,
                canny_aperture=3,
                mode='canny_sobel'):
        """
        Processes an image using Canny, Sobel, or both combined, depending on the mode.
        
        :param image: Input image
        :param ksize: Kernel size for the Sobel operator
        :param canny_low: Lower threshold for the Canny edge detector
        :param canny_high: Higher threshold for the Canny edge detector
        :param canny_aperture: Aperture size for the Canny operator
        :param mode: Processing mode ('canny', 'sobel', or 'canny_sobel')
        :return: Processed image magnitude
        """
        # Validate parameters (could be expanded based on specific requirements)
        assert mode in ['canny', 'sobel', 'canny_sobel'], "Invalid mode specified."
        assert ksize in [1, 3, 5, 7], "Invalid ksize specified. Must be 1, 3, 5, or 7."
        assert canny_low < canny_high, "canny_low must be less than canny_high."
        assert canny_aperture in [3, 5, 7], "Invalid canny_aperture specified. Must be 3, 5, or 7."
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 5, 75, 75)
        
        if mode == "canny_sobel":
            edges = cv2.Canny(blur, canny_low, canny_high, apertureSize=canny_aperture)
            sobelx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=ksize)
            sobely = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=ksize)
            magnitude, _ = cv2.cartToPolar(sobelx, sobely)
        elif mode == "canny":
            magnitude = cv2.Canny(blur, canny_low, canny_high, apertureSize=canny_aperture)
        else:  # Sobel mode
            sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=ksize)
            sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=ksize)
            magnitude, _ = cv2.cartToPolar(sobelx, sobely)

        return magnitude
