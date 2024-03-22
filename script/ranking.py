import json
# Assuming test_results is a dictionary containing the evaluation results
with open("test_results.json", "r") as f:
    test_results = json.load(f)
# Define a function to calculate the overall score for each configuration
def calculate_score(result):
    return result["Accuracy"] + result["Precision"] + result["Recall"] + result["F1-Score"]

# Calculate the overall score for each configuration
for config_id, result in test_results.items():
    score = calculate_score(result)
    test_results[config_id]["Overall Score"] = score

# Rank configurations based on the overall score
ranked_results = sorted(test_results.items(), key=lambda x: x[1]["Overall Score"], reverse=True)

top_10_results = ranked_results[:10]

# Print the top 10 configurations
print("Top 10 configurations:")
for rank, (config_id, result) in enumerate(top_10_results, start=1):
    print(f"Rank {rank}: Configuration {config_id}, Overall Score: {result['Overall Score']}")


import csv

# Define the file path for saving the CSV file
csv_file = "result_rangking.csv"

# Define the field names for the CSV file
field_names = ["Rank", "Configuration ID", "canny_aperture", "canny_high", "canny_low", "ksize", "mode", "Accuracy", "Precision", "Recall", "F1-Score", "Training Time (s)", "Max Memory Usage (MB)"]

# Write the top 10 configurations to the CSV file
with open(csv_file, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=field_names)
    
    # Write the header row
    writer.writeheader()
    
    # Write the data for each configuration
    for rank, (config_id, result) in enumerate(ranked_results, start=1):
        writer.writerow({
            "Rank": rank,
            "Configuration ID": config_id,
            "canny_aperture": result["Config"]["canny_aperture"],
            "canny_high": result["Config"]["canny_high"],
            "canny_low": result["Config"]["canny_low"],
            "ksize": result["Config"]["ksize"],
            "mode": result["Config"]["mode"],
            "Accuracy": result["Accuracy"],
            "Precision": result["Precision"],
            "Recall": result["Recall"],
            "F1-Score": result["F1-Score"],
            "Training Time (s)": result["Training Time (s)"],
            "Max Memory Usage (MB)": result["Max Memory Usage (MB)"]
        })

print(f"CSV file '{csv_file}' has been saved successfully.")