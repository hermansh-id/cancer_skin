import json

# Parameters
models = ['canny', 'sobel', 'canny_sobel']
sobel_ksize = [1, 3, 5]
canny_low_thresholds = [50, 100, 150]
canny_high_thresholds = [100, 200, 250]
aperture_sizes = [3, 5, 7]

# Generate experiment variations
experiments = []
experiment_id = 1

for model in models:
    for ksize in sobel_ksize:
        for low_threshold in canny_low_thresholds:
            for high_threshold in canny_high_thresholds:
                if high_threshold <= low_threshold:
                    continue
                for aperture_size in aperture_sizes:
                    # if model == 'Canny':
                    #     experiments.append({
                    #         "Experiment ID": experiment_id,
                    #         "Model": model,
                    #         "Canny Low Threshold": low_threshold,
                    #         "Canny High Threshold": high_threshold
                    #     })
                    # elif model == 'Sobel':
                    #     experiments.append({
                    #         "Experiment ID": experiment_id,
                    #         "Model": model,
                    #         "Sobel Ksize": ksize,
                    #         "ApertureSize": aperture_size
                    #     })
                    # else:  # Canny+Sobel combines both
                    
                    experiments.append({
                        "mode": model,
                        "ksize": ksize,
                        "canny_aperture": aperture_size,
                        "canny_low": low_threshold,
                        "canny_high": high_threshold
                    })

def remove_duplicates(list_of_dicts):
    # Convert each dictionary to a tuple of its items for hashability
    tuple_set = set(tuple(sorted(d.items())) for d in list_of_dicts)
    # Convert each tuple back to a dictionary
    unique_dicts = [dict(t) for t in tuple_set]
    return unique_dicts

unique_data = remove_duplicates(experiments)

unique_data = [{"id": i, **data} for i, data in enumerate(unique_data, start=1)]

# Convert to JSON
experiments_json = json.dumps(unique_data, indent=4)

# Output to file (or you can directly use it)
with open('experiment_configurations.json', 'w') as file:
    file.write(experiments_json)

print("JSON file with experiment configurations has been created.")
