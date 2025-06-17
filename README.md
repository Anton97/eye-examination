# Pupil Analysis Package

This package provides functions for analyzing video files to determine pupil diameters and for machine learning tasks related to pupil data, saving the results in CSV format.

## Installation

To install the `pupil_analysis_package`, navigate to the root directory of the package (where `setup.py` is located) and run:

```bash
pip install .
```

This will install the package and its dependencies (`opencv-python`, `numpy`, `pandas`, `tqdm`, `scikit-learn`, `joblib`).

## Usage

The package contains functions in `analysis_module.py` for video processing and `ml_module.py` for machine learning tasks.

### Video Analysis Functions (`analysis_module.py`)

#### `process_video(video_path, output_dir)`

Processes a single video and saves the pupil analysis results to a CSV file.

**Parameters:**

*   `video_path` (str): The full path to the video file to be processed.
*   `output_dir` (str): The directory where the results (CSV file) will be saved.

**Returns:**

*   `pandas.DataFrame`: A DataFrame containing the analysis results (time, left pupil diameter, right pupil diameter), or `None` in case of an error.

**Example Usage:**

```python
from pupil_analysis_package.analysis_module import process_video

video_file = "path/to/your/video.mp4"
results_directory = "path/to/your/results"

df_results = process_video(video_file, results_directory)

if df_results is not None:
    print("Video processed successfully.")
    print(df_results.head())
else:
    print("An error occurred during video processing.")
```

#### `process_videos_in_folder_recursive(input_root, output_root)`

Recursively processes all video files within a specified root directory and saves the results in a similar folder structure.

**Parameters:**

*   `input_root` (str): The root directory containing video files for processing.
*   `output_root` (str): The root directory where the results will be saved. The folder structure within `output_root` will mirror that of `input_root`.

**Returns:**

*   `dict`: A dictionary where keys are the full paths to the processed video files, and values are the corresponding `pandas.DataFrame` objects with the results.

**Example Usage:**

```python
from pupil_analysis_package.analysis_module import process_videos_in_folder_recursive

input_folder = "path/to/your/videos_folder"
output_folder = "path/to/your/output_results_folder"

all_results = process_videos_in_folder_recursive(input_folder, output_folder)

for video_path, df_results in all_results.items():
    print(f"Processed video: {video_path}")
    print(df_results.head())
```

### Machine Learning Functions (`ml_module.py`)

#### `load_labeled_csvs(base_folder)`

Loads and concatenates labeled CSV files from a specified base folder. It expects subfolders named "норма" (normal) and "отклонение" (deviation) containing CSV files.

**Parameters:**

*   `base_folder` (str): The base directory containing the labeled subfolders.

**Returns:**

*   `pandas.DataFrame`: A concatenated DataFrame with an added 'label' column (0 for 'норма', 1 for 'отклонение').

#### `augment_dataframe(df, n_augments=2, noise_std=0.05)`

Augments a DataFrame by adding Gaussian noise to specified features. Useful for increasing dataset size for training.

**Parameters:**

*   `df` (pandas.DataFrame): The input DataFrame to augment.
*   `n_augments` (int): The number of augmented copies to create.
*   `noise_std` (float): The standard deviation of the Gaussian noise as a fraction of the feature's standard deviation.

**Returns:**

*   `pandas.DataFrame`: The original DataFrame concatenated with its augmented copies.

#### `train_and_save_model(train_data_path, val_data_path, model_save_path="best_elasticnet_model.pkl")`

Trains an ElasticNet model using data from specified training and validation paths and saves the best model.

**Parameters:**

*   `train_data_path` (str): Path to the training data folder (containing 'норма' and 'отклонение' subfolders).
*   `val_data_path` (str): Path to the validation data folder (containing 'норма' and 'отклонение' subfolders).
*   `model_save_path` (str): (Optional) Path to save the trained model. Defaults to "best_elasticnet_model.pkl".

#### `predict_with_model(video_results_df, model_path="best_elasticnet_model.pkl")`

Makes predictions using a pre-trained ElasticNet model on new video analysis results.

**Parameters:**

*   `video_results_df` (pandas.DataFrame): DataFrame containing pupil analysis results (e.g., from `process_video`).
*   `model_path` (str): (Optional) Path to the pre-trained model. Defaults to "best_elasticnet_model.pkl".

**Returns:**

*   `numpy.ndarray`: An array of predictions (labels).

## Data Structure

For video analysis, the script expects input video files to be located in any subfolder within `input_root`. The processing results will be saved in `output_root` while maintaining the original folder structure.

For machine learning, `load_labeled_csvs` expects a structure like:

```
base_folder/
├── норма/
│   └── video1_results.csv
│   └── video2_results.csv
└── отклонение/
    └── video3_results.csv
    └── video4_results.csv
```

## Notes

*   The `process_video` and `process_videos_in_folder_recursive` functions save data to the CSV file only for frames where both the left and right pupils were detected simultaneously.
*   The `tqdm` library is used to display processing progress.
*   The `predict_with_model` function assumes the scaler used during training is implicitly handled. For robust production use, the `StandardScaler` object should also be saved and loaded alongside the model.


