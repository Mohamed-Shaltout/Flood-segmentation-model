# Water Segmentation Model

## Description

This project implements a deep learning model for segmenting water in remote sensing or satellite images. The model uses a U-Net architecture built with TensorFlow and Keras, optimized for pixel-wise segmentation of water bodies from multi-channel TIFF images.

### Key Features

- **Input Data**: Supports multi-channel TIFF images (such as satellite or aerial imagery) and corresponding mask labels for water presence.
- **Data Handling**: Employs a custom data generator to efficiently read, preprocess, and batch images and masks.
- **Neural Network**: Uses a U-Net convolutional neural network for semantic segmentation.
- **Training & Evaluation**: Includes training procedures with accuracy and binary cross-entropy loss tracking, plus robust evaluation on test data.
- **Performance**: Achieves over 92% test accuracy in example settings.

## Project Structure

```
watersegmentation-model-1.ipynb     # Main Jupyter notebook with code, explanations, and results
/inputs
    /images                        # Directory for input satellite/aerial images (.tif)
    /labels                        # Directory for ground truth segmentation masks
/outputs
    model.h5                       # (Optional) Saved trained model weights
```

## Requirements

- Python 3.x
- TensorFlow
- Keras
- OpenCV
- Pillow
- numpy
- tifffile
- matplotlib

Install dependencies with:

```
pip install tensorflow keras opencv-python pillow numpy tifffile matplotlib
```

## Usage

### 1. Prepare Data

- Place multi-channel input images in the `/inputs/images` folder.
- Place corresponding binary mask labels in `/inputs/labels`, ensuring filenames match input images.

### 2. Configure and Run the Model

Open the `watersegmentation-model-1.ipynb` notebook and run the cells in order:

- **Data Generator**: Reads and preprocesses batches of images and masks.
- **Model Building**: Constructs the U-Net architecture.
- **Training**: Fits the model with training and validation data, showing real-time progress.
- **Evaluation**: Tests the model on held-out data, displaying accuracy and loss.

### 3. Interpretation

- Visualize predictions and compare with true masks using built-in visualization sections.
- Evaluate metrics (accuracy, loss).

### 4. Inference

- For new images, adapt the data generator for your test images and use `model.predict()` to generate segmentation masks.

## Example Results

| Metric        | Value    |
|---------------|----------|
| Test Accuracy | 92.17%   |
| Test Loss     | 0.2242   |

*Results may vary according to your data and configuration.*

## Customization

- **Adding Channels**: Change the input shape in the data generator and model for your imagery’s bands.
- **Modifying Architecture**: Adjust layers, filters, or activation functions as needed.
- **Loss Functions**: Switch from binary cross-entropy to another loss function to suit your task.

## Troubleshooting

- **CUDA/Driver Errors**: If not using GPU or facing CUDA errors, switch to CPU mode.
- **Out-Of-Memory**: Reduce batch size or image dimensions if memory issues arise.
- **Data Shapes**: Ensure both images and masks have correct, matching shape and channel counts.

## License

This project is released for educational and research purposes. It can be adapted and used as needed.

This model offers a strong foundation for water segmentation in multi-channel imagery and can be extended for broader remote sensing or geospatial tasks.
