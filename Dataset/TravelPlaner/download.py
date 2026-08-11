from datasets import load_dataset

DATASET_NAME = "osunlp/TravelPlanner"
OUTPUTS = {
    "train": "TravelPlanner_train.csv",
    "validation": "TravelPlanner_val.csv",
    "test": "TravelPlanner_test.csv",
}

for config, output_path in OUTPUTS.items():
    dataset = load_dataset(DATASET_NAME, config)
    split = dataset[config] if config in dataset else next(iter(dataset.values()))
    split.to_csv(output_path, index=False)
    print(f"Saved {config}: {len(split)} rows -> {output_path}")
