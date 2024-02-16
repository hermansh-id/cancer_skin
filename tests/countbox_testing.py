import unittest
from counter_box.predict import Predict

class TestCountBox(unittest.TestCase):
    
    def test_valid_image_file(self):
        image_file = "test.jpg"
        # Create an instance of the Predict class
        predict = Predict()
        # Call the countbox function with the valid image file
        predict.countbox(image_file)
        # Assert that the plot is displayed properly
        # (This assertion might vary based on the specific testing environment)
        
if __name__ == '__main__':
    unittest.main()