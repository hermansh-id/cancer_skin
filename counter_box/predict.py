from counter_box.preprocess import Preprocess
from counter_box.countbox import CountBox
from matplotlib import pyplot as plt
import numpy as np

class Predict:
    def __init__(self):
        self.preprocess = Preprocess()
        self.cb = CountBox()

    def countbox(self, image_file):
        image = self.preprocess.process(image_file)
        slope, intercept, sizes = self.cb.hausdorff(image)
        
        plt.loglog(1.0 / sizes, np.exp(intercept) * (1.0 / sizes) ** slope, '-')
        plt.xlabel('log(1 / size)')
        plt.ylabel('log(count)')
        plt.title('Hausdorff Fractal Dimension: {:.3f}'.format(slope))
        # Show the plot
        plt.show()