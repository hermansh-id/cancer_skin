import json
import time
import psutil  # For monitoring memory usage
from memory_profiler import memory_usage  # For monitoring memory usage
from counter_box.classifier import SkinCancerClassifier
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
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

# Concatenate and create labels
X_train = np.concatenate((X_benign_train, X_malignant_train), axis=0)
y_train = np.concatenate((np.zeros(X_benign_train.shape[0]), np.ones(X_malignant_train.shape[0])), axis=0)


# Split into train and test sets using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_size_ratio, random_state=42)


# Check if experiment configurations JSON file exists
if os.path.exists(test_results_file):
    with open(test_results_file, 'r') as f:
        experiment_results = json.load(f)
else:
    experiment_results = {}

with open(experiment_config_file, 'r') as f:
    experiment_configs = json.load(f)

# Get list of experiment results keys
experiment_result_id = [int(i) for i in list(experiment_results.keys())]
# experiment_result_id = [int(config['id']) for config in experiment_results.values()]

print("++++ START THE EXPERIMENTS ++++")
# Initialize dictionary to store test 


len_experiment = len(experiment_configs)
last_time = 0
# Iterate over experiment configurations
for config in experiment_configs:
    try:
    # Initialize classifier
        classifier.reset()

        # Check if the configuration has already been processed
        config_id = config['id']
        del config['id']
        if config_id in experiment_result_id:
            print(f"Skipping Experiment {config_id} (already processed)...")
            continue
        
        print("="*20)
        print(f"Running Experiment {config_id}/{len_experiment}...")
        print("CONFIG:")
        for key, value in config.items():
            print(f"{key}: {value}")
        
        # Start measuring training time
        start_time = time.time()
        
        # Start monitoring memory usage
        mem_usage_before = memory_usage(-1, interval=1)
        print("\nTRAINING MODEL:")
        # Train classifier with current configuration
        classifier.train(X_train, y_train, **config)
        
        # Stop measuring training time
        end_time = time.time()
        training_time = end_time - start_time
        
        # Stop monitoring memory usage
        mem_usage_after = memory_usage(-1, interval=1)
        max_memory_usage = max(mem_usage_after) - max(mem_usage_before)
        
        # Predict on test set
        print("\nTESTING OUT:")
        print(len(X_test))
        y_pred, nan_indices = classifier.predict(X_test, **config)
        print(len(y_pred), len(y_test), len(nan_indices))
        tmp_y_test = y_test[~nan_indices]
        # y_test = y_test[~nan_indices]

        # Calculate evaluation metrics
        accuracy = accuracy_score(tmp_y_test, y_pred)
        precision = precision_score(tmp_y_test, y_pred)
        recall = recall_score(tmp_y_test, y_pred)
        f1 = f1_score(tmp_y_test, y_pred)

        confusion = confusion_matrix(tmp_y_test, y_pred)

        # Store test results
        results = {
            "Config": config,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Confusion Matrix": confusion.tolist(),
            "Training Time (s)": training_time,
            "Max Memory Usage (MB)": max_memory_usage
        }
        print("\nRESULT:")
        for key, value in results.items():
            print(f"{key}: {value}")
        experiment_results[config_id] = results
        # Save test results to JSON after each iteration
        with open(test_results_file, 'w') as f:
            json.dump(experiment_results, f, indent=4)

        print("Test results have been saved to", test_results_file)
        est = training_time * (len(experiment_configs) - config_id)
        print(f"Estimated time remaining: {est} seconds/ {est / 60} minutes")
        print("="*20)
    except Exception as e:
        print(f"Skipping Experiment {config_id}/{len_experiment}...")
        print(f"Error: {e}")
        with open("error_log.txt", "a") as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
            f.write(f"Error: {e}\n")
            
        continue

# Save all experiment results to JSON
# with open(test_results, 'w') as f:
#     json.dump(test_results, f, indent=4)

# print("Experiment results have been saved to", experiment_config_file)
