# **Vegetable Image Classification Using CNN**

This project focuses on building and training a Convolutional Neural Network (CNN) to classify images of vegetables into 15 distinct categories using TensorFlow and Keras. The model is trained on a custom dataset of vegetable images, with preprocessing, training, validation, and visualization of results.

---

## **Table of Contents**
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Dependencies](#dependencies)
- [How to Run the Project](#how-to-run-the-project)
- [Results](#results)
- [File Structure](#file-structure)
- [Future Improvements](#future-improvements)

---

## **Project Overview**
The aim of this project is to classify vegetable images into their respective categories using deep learning. This solution could be useful in applications such as automated vegetable recognition for retail, agriculture, or inventory systems.

---

## **Dataset**
- **Source**: The dataset contains two folders:
  - **Training Set**: Images for training the model.
  - **Validation Set**: Images for validating the model.
- **Categories**: 15 different vegetable classes.
- **Preprocessing**:
  - Images resized to **150x150 pixels**.
  - Images normalized for input into the CNN.

---

## **Model Architecture**
The CNN model consists of:
1. **Convolutional Layers**:
   - Extract features using 32 and 64 filters with a kernel size of 3x3.
   - `ReLU` activation is used for non-linearity.
2. **MaxPooling Layers**:
   - Reduce spatial dimensions to prevent overfitting.
3. **Dropout Layers**:
   - Regularization to avoid overfitting.
4. **Dense Layers**:
   - Fully connected layers with 512 and 256 neurons for feature learning.
5. **Output Layer**:
   - A softmax layer with 15 units for multi-class classification.

---

## **Dependencies**
The project requires the following Python libraries:
- `tensorflow`
- `matplotlib`
- `json`

You can install them using:
```bash
pip install tensorflow matplotlib
```

---

## **How to Run the Project**
1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Place the dataset in the directory `./Vegetable_Images/` with subfolders `train` and `validation`.
3. Run the script:
   ```bash
   python vegetable_classification.py
   ```
4. The model will be trained for 28 epochs, and the training history will be saved as `training_hist.json`. The trained model will be saved as `trained_model.h5`.

---

## **Results**
The following plots visualize the training and validation performance:
1. **Training Accuracy vs Epochs**:
   ![Training Accuracy](training_accuracy_plot.png)

2. **Validation Accuracy vs Epochs**:
   ![Validation Accuracy](validation_accuracy_plot.png)

Metrics:
- **Training Accuracy**: X%
- **Validation Accuracy**: Y%

---

## **File Structure**
```
Vegetable_Classification/
│
├── Vegetable_Images/
│   ├── train/          # Training images
│   ├── validation/     # Validation images
│
├── trained_model.h5    # Saved model
├── training_hist.json  # Training history
├── vegetable_classification.py # Main script
├── README.md           # Project documentation
```

---

## **Future Improvements**
- Add more data augmentation techniques to improve generalization.
- Experiment with different architectures like ResNet or MobileNet for better accuracy.
- Fine-tune the model on a pre-trained network for faster training and improved performance.
- Deploy the trained model as a web or mobile application for real-world use.

---

## **License**
This project is for educational purposes. Feel free to modify and adapt for your use cases.

--- 

Let me know if you'd like further customizations!
