import os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from counter_box.countbox import CountBox
from counter_box.preprocess import Preprocess
from tqdm import tqdm
import pickle
# Import CountBox class and any other necessary modules

class SkinCancerClassifier:
    def __init__(self):
        self.model = RandomForestClassifier()
        self.cb = CountBox()
        self.pr = Preprocess()

    def reset(self):
        self.model = RandomForestClassifier()
        
    def read_images(self, folder_path):
        read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
        images = [read(os.path.join(folder_path, filename)) for filename in os.listdir(folder_path)]
        return np.array(images, dtype='uint8')

    def calculate_hausdorff(self, X, **kwargs):
        hausdorff_dimensions = []
        for img in tqdm(X):
            preprocessed_img = self.pr.process(img, **kwargs)
            slope, intercept, _ = self.cb.hausdorff(preprocessed_img)
            hausdorff_dimensions.append(slope)
        return np.array(hausdorff_dimensions).reshape(-1, 1)

    def train(self, 
              X_train, 
              y_train,
              **kwargs
              ):
            X_train_hausdorff = self.calculate_hausdorff(X_train, **kwargs)
            nan_indices = np.isnan(X_train_hausdorff).any(axis=1)
            X_train = X_train[~nan_indices]
            y_train = y_train[~nan_indices]
            
            # print("Number of NaN values in X_train_hausdorff:", nan_indices.sum())
            X_train_hausdorff = X_train_hausdorff[~nan_indices]
            # print("NUmber of data points in X_train_hausdorff:", X_train_hausdorff.shape[0])
            X_train_combined = np.hstack((X_train.reshape(X_train.shape[0], -1), X_train_hausdorff))
            self.model.fit(X_train_combined, y_train)

    def predict(self, X_test, **kwargs):
        X_test_hausdorff = self.calculate_hausdorff(X_test, **kwargs)
        nan_indices = np.isnan(X_test_hausdorff).any(axis=1)
        X_test = X_test[~nan_indices]
        X_test_hausdorff = X_test_hausdorff[~nan_indices]
        X_test_combined = np.hstack((X_test.reshape(X_test.shape[0], -1), X_test_hausdorff))
        return self.model.predict(X_test_combined), nan_indices

    def evaluate(self, X_test, y_test, **kwargs):
        y_pred, nan_indices = self.predict(X_test, **kwargs)
        y_test = y_test[~nan_indices]
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)
    
    def load_model(self, filename):
        with open(filename, 'rb') as file:
            self.model = pickle.load(file)