# Apple Disease Detection using Hybrid VGG-19

## Abstract
### Motivation
Plant disease detection systems play a crucial role in early identification, preventing significant crop losses, and ensuring optimal productivity. According to India's Commerce Ministry, apple exports have surged over 82% since 2014, with major production states being:
- **Jammu & Kashmir (75%)**
- **Himachal Pradesh (20%)**
- **Uttarakhand (2.5%)**

However, apple orchards face increasing threats from diseases such as **apple scab, cedar-apple rust, and black rot**, affecting yield and export quality. This highlights the necessity of an efficient apple disease detection system to assist farmers in improving crop quality and yield.

### Problem Gap
Despite advancements in plant disease detection using **machine learning** and **deep learning**, existing models lack accuracy, generalizability, and scalability in real-world agricultural settings. Many models in India still depend on **traditional convolutional networks**, which may be inefficient.

### Contribution
This work presents a **Novel Hybrid VGG-19 model**, leveraging:
- **VGG-19 base architecture**
- **Depthwise separable convolution**
- **Residual network identity block**

## Literature Review
Several studies have explored deep learning techniques for plant disease detection:
1. **Kotwal J. et al.**: Transitioned from manual feature extraction to CNN, achieving high accuracy using Grad-CAM for visualization.
2. **Chohan M. et al.**: Achieved **98.3% accuracy** using a CNN model on the Plant Village dataset.
3. **Shelar N. et al.**: Used **VGG19** to attain **95.6% accuracy**, developing an Android app for real-time detection.
4. **Hassan S. et al.**: Combined **Inception and Residual connections**, reducing model parameters by **70%** while maintaining accuracy.
5. **Peyal H. et al.**: Developed a lightweight CNN outperforming VGG16 and InceptionV3, achieving **97.36% accuracy**.

## Methods
### Convolutional Neural Networks (CNN)
CNNs excel in image classification by automatically extracting features from training data. The process includes:
1. **Convolutional Layer**: Extracts spatial features using a sliding kernel.
2. **Pooling Layer**: Reduces dimensionality while retaining key features.
3. **Flattening Layer**: Converts feature maps into a 1D array.
4. **Fully Connected Layers**: Perform classification.

### Residual Network (ResNet)
ResNet introduces **skip connections** to tackle the **vanishing gradient problem**.
- Formula: \( h(x) = f(x) + x \)
- Skip connections improve gradient flow, enhancing deep network training.

### Depthwise Separable Convolution
Decomposes standard convolution into two steps:
1. **Depthwise Convolution**: Applies a single filter per channel.
2. **Pointwise Convolution**: Uses **1×1** convolution to combine outputs.
- Reduces computational cost while maintaining accuracy.

### Visual Geometry Group-19 (VGG-19)
VGG-19 consists of **16 convolutional layers** and **3 fully connected layers**, using:
- **3×3 filters**
- **Max pooling (2×2)**
- **Softmax activation** for classification.

## Proposed Model
The proposed model integrates:
- **VGG-19 as base architecture**
- **Depthwise separable convolutions** (reducing computational complexity)
- **Residual blocks** (ensuring better gradient flow)

### Residual Block
- Incorporates **two depthwise convolution layers** followed by **pointwise convolution**.
- Uses **L2 regularization and ReLU activation**.
- Shortcut connections improve training efficiency.

#### ![Residual Block](Fig.3.png)

### Main Function
- Input: **224×224×3** images.
- **First Conv Block**: 2 **depthwise convolutions** (64 filters) → **pointwise convolution** (64 filters) → **Max pooling**.
- **Blocks 2-5**: Increase filter count from **128 to 512** with residual connections.
- Fully connected layers: **4096 neurons**, followed by dropout (**0.2** rate).
- **Output layer**: Number of neurons = **number of classes**.

#### ![Hybrid VGG-19 Architecture](Fig.4.png)

## Dataset
The **New Plant Disease Dataset (PlantVillage)** was used:
- **4 Classes**: Apple Scab, Apple Black Rot, Apple Cedar Rust, Apple Healthy.
- **Train-Test Split**: **80:20**.

#### ![Apple Scab](Fig.5.png)
#### ![Apple Black Rot](Fig.6.png)
#### ![Apple Cedar Rust](Fig.7.png)
#### ![Apple Healthy](Fig.8.png)

### Pre-processing
- **Background Removal** using `rembg` (U²-Net deep learning model).

## Experiment and Results
### Training Results
- **Accuracy**: Increased from **0.5683** → **0.9831** in 49 epochs.
- **Validation Accuracy**: Peaked at **0.9811**.

### Evaluation
Test Set 1:
| Class             | Precision | Recall | F1 Score | Samples |
|-------------------|-----------|--------|----------|---------|
| Apple Scab        | 0.97      | 0.97   | 0.97     | 336     |
| Apple Black Rot   | 0.98      | 0.97   | 0.98     | 376     |
| Apple Cedar Rust  | 0.97      | 0.99   | 0.98     | 349     |
| Apple Healthy     | 0.98      | 0.97   | 0.98     | 346     |

Test Set 2:
| Class             | Precision | Recall | F1 Score | Samples |
|-------------------|-----------|--------|----------|---------|
| Apple Scab        | 0.78      | 0.60   | 0.68     | 281     |
| Apple Black Rot   | 0.74      | 0.98   | 0.84     | 266     |
| Apple Cedar Rust  | 0.65      | 0.99   | 0.79     | 208     |
| Apple Healthy     | 0.92      | 0.45   | 0.61     | 255     |

#### ![Training Loss vs Validation Loss](Fig.9.png)
#### ![Training Accuracy vs Validation Accuracy](Fig.10.png)

### Performance Metrics
- **Accuracy**: \( \frac{TP + TN}{TP + TN + FP + FN} \)
- **Precision**: \( \frac{TP}{TP + FP} \)
- **Recall**: \( \frac{TP}{TP + FN} \)
- **F1 Score**: \( 2 \times \frac{Precision \times Recall}{Precision + Recall} \)

### GRAD-CAM Visualization
Grad-CAM highlights the model's decision-making process, showing regions most influential in classification.

## Conclusion
- The **Hybrid VGG-19 model** efficiently detects apple diseases.
- **Depthwise separable convolution** enhances computational efficiency.
- **Residual connections** improve training stability.
- Achieved **97.65% accuracy** on test set 1 and **74.46% accuracy** on test set 2.


## License
This project is licensed under the MIT License.

---
### 📌 Feel free to contribute and improve the repository!
