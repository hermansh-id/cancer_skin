from counter_box.classifier import SkinCancerClassifier
import numpy as np

classifier = SkinCancerClassifier()

folder_benign_test = '../data/test/benign'
folder_malignant_test = '../data/test/malignant'

classifier.load_model('skin_cancer_model.pkl')

X_benign_test = classifier.read_images(folder_benign_test)
X_malignant_test = classifier.read_images(folder_malignant_test)

X_test = np.concatenate((X_benign_test, X_malignant_test), axis=0)
y_test = np.concatenate((np.zeros(X_benign_test.shape[0]), np.ones(X_malignant_test.shape[0])), axis=0)

classifier.evaluate(X_test, y_test)
