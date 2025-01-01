# Age and Gender Prediction Using Convolutional Neural Networks

This project involves building a Convolutional Neural Network (CNN) model using TensorFlow/Keras to predict age and gender from the UTKFace dataset. The model has dual outputs: one for gender classification and another for age regression.

## Features

- **Dataset Download**: Downloads the UTKFace dataset using KaggleHub.
- **Preprocessing**: Resizes images, normalizes pixel values, and prepares data for training.
- **Model Architecture**: Builds a CNN with separate outputs for age regression and gender classification.
- **Training and Evaluation**: Trains the model and evaluates its performance on the test set.

## Requirements

- Python 3.7+
- Required Python packages:
  - TensorFlow
  - Keras
  - NumPy
  - Pandas
  - Matplotlib
  - Seaborn
  - scikit-learn
  - PIL (Pillow)
  - KaggleHub

Install the dependencies using:
```bash
pip install -r requirements.txt
```

## Code Breakdown

### Importing Libraries
The necessary libraries for data loading, preprocessing, model building, and visualization are imported. KaggleHub is used to download the dataset.

### Dataset Preparation

#### Dataset Download
The UTKFace dataset is downloaded using KaggleHub:
```python
import kagglehub
jangedoo_utkface_new_path = kagglehub.dataset_download('jangedoo/utkface-new')
```

#### Loading Data
Images, ages, and genders are loaded from the dataset:
- Images are resized to 128x128 pixels.
- Ages and genders are extracted from filenames.
- Images are normalized to the range [0, 1].

### Data Splitting
The dataset is split into training and test sets:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_gender_train, y_gender_test, y_age_train, y_age_test = train_test_split(
    images, genders, ages, test_size=0.2, random_state=42)
```

### Model Architecture
A CNN is built using TensorFlow/Keras with the following components:

#### Input Layer
```python
input_size = (128, 128, 3)  # RGB images
inputs = Input(input_size)
```

#### Convolutional Layers
Three convolutional blocks with ReLU activation, He uniform kernel initializer, and max-pooling:
```python
X = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(inputs)
X = MaxPooling2D((2, 2))(X)
...
```

#### Fully Connected Layers
Flatten the output and add a dense layer with dropout for regularization:
```python
X = Flatten()(X)
dense_1 = Dense(256, activation='relu')(X)
dropout_1 = Dropout(0.5)(dense_1)
```

#### Outputs
- **Gender Output**: Sigmoid activation for binary classification.
- **Age Output**: ReLU activation for regression.
```python
output_1 = Dense(1, activation='sigmoid', name='gender_output')(dropout_1)
output_2 = Dense(1, activation='relu', name='age_output')(dropout_1)
```

#### Model Compilation
The model is compiled with the Adam optimizer and different losses for the outputs:
```python
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss={'gender_output': 'binary_crossentropy', 'age_output': 'mean_squared_error'},
              metrics={'gender_output': 'accuracy', 'age_output': 'mae'})
```

### Model Training
The model is trained for one epoch (can be increased for better results):
```python
history = model.fit(X_train,
                    {'gender_output': y_gender_train, 'age_output': y_age_train},
                    validation_data=(X_test, {'gender_output': y_gender_test, 'age_output': y_age_test}),
                    epochs=1,
                    batch_size=128)
```

### Model Evaluation
The model is evaluated on the test set, and results are printed:
```python
results = model.evaluate(X_test,
                         {'gender_output': y_gender_test, 'age_output': y_age_test})

print("Test Loss:", results[0])
print("Gender Output Loss:", results[1])
print("Age Output Loss:", results[2])
print("Test Gender Accuracy:", results[3])
print("Test Age MAE:", results[4])
```

## Insights and Next Steps

- **Performance Metrics**: Gender accuracy and age MAE are key metrics to evaluate the model.
- **Visualization**: Plotting loss and accuracy curves can help identify overfitting or underfitting.
- **Enhancements**:
  - Use more epochs and tune hyperparameters for improved results.
  - Add data augmentation to improve model robustness.
  - Explore advanced architectures like ResNet or EfficientNet for better accuracy.

## License

This project is licensed under the MIT License. Feel free to use and modify it as per your requirements.

