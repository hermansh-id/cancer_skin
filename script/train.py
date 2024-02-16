from counter_box.classifier import SkinCancerClassifier
import numpy as np

classifier = SkinCancerClassifier()

folder_benign_train = '../data/train/benign'
folder_malignant_train = '../data/train/malignant'


X_benign_train = classifier.read_images(folder_benign_train)
X_malignant_train = classifier.read_images(folder_malignant_train)

X_train = np.concatenate((X_benign_train, X_malignant_train), axis=0)
y_train = np.concatenate((np.zeros(X_benign_train.shape[0]), np.ones(X_malignant_train.shape[0])), axis=0)

classifier.train(X_train, y_train)

classifier.save_model('skin_cancer_model.pkl')

# Evaluate the model

