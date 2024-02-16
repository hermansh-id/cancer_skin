import cv2

class Preprocess:

    def process(self, image):
        # image = cv2.imread(image_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 5, 75, 75)
        edges = cv2.Canny(blur, 100, 200)
        sobelx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=5)
        # return sobelx + sobely
        magnitude, _ = cv2.cartToPolar(sobelx, sobely)
        return magnitude