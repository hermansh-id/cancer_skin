from memory_profiler import memory_usage  # For monitoring memory usage
from counter_box.classifier import SkinCancerClassifier
import numpy as np
from sklearn.model_selection import train_test_split  # Importing train_test_split function

# Define parameters
max_training_images = 200
test_size_ratio = 0.1
experiment_config_file = 'experiment_configurations.json'
test_results_file = 'test_results.json'

# Initialize classifier
classifier = SkinCancerClassifier()

# Read training data
folder_benign_train = '../data/train/benign'
folder_malignant_train = '../data/train/malignant'
X_benign_train = classifier.read_images(folder_benign_train)[:max_training_images // 2]
X_malignant_train = classifier.read_images(folder_malignant_train)[:max_training_images // 2]

print(len(X_benign_train))
print(len(X_malignant_train))

# Concatenate and create labels
X_train = np.concatenate((X_benign_train, X_malignant_train), axis=0)
y_train = np.concatenate((np.zeros(X_benign_train.shape[0]), np.ones(X_malignant_train.shape[0])), axis=0)


# Split into train and test sets using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_size_ratio, random_state=42)

print(y_train)
print(y_test)
print(f"Length of data for training: {len(X_train)}")
print(f"Length of data for testing: {len(X_test)}")
