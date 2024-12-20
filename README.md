### **README: Retinal Disease Classifier**

---

# **Retinal Disease Classifier**

A deep learning-based project for automated classification of retinal fundus images into 11 disease categories, using state-of-the-art architectures like EfficientNet-B0. This project incorporates Focal Loss to address class imbalance, data augmentation for robust training, and comprehensive evaluation metrics for detailed insights.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Setup Instructions](#setup-instructions)
3. [How to Run](#how-to-run)
4. [Expected Output](#expected-output)
5. [Pre-trained Model](#pre-trained-model)
6. [Acknowledgments](#acknowledgments)

---

## **Project Overview**
This project aims to classify retinal fundus images into 11 categories:
- Dry AMD
- Wet AMD
- Glaucoma
- Diabetic Retinopathy (Mild, Moderate, Severe, Proliferative)
- Cataract
- Hypertensive Retinopathy
- Pathological Myopia
- Normal Fundus

EfficientNet-B0 with transfer learning and Focal Loss is used to handle the significant class imbalance. The model has been fine-tuned for optimal accuracy and evaluated on a diverse dataset of retinal fundus images.

---

## **Setup Instructions**
### **1. Clone the Repository**
```bash
git clone https://github.com/username/retinal-disease-classifier.git
cd retinal-disease-classifier
```

### **2. Install Dependencies**
Use the following command to install required Python libraries:
```bash
pip install -r requirements.txt
```

Alternatively, for Conda users:
```bash
conda env create -f conda.yml
conda activate retinal-disease-classifier
```

### **3. Dataset Setup**
Place your dataset in the following structure:
```
Retinal Fundus Images/
├── train/
├── val/
├── test/
```
Ensure the dataset paths are updated in `src/config.py` if necessary.

---

## **How to Run**
### **1. Train the Model**
Run the training script to train EfficientNet-B0:
```bash
python src/main.py
```

### **2. Test and Demo Script**
Use the demo.ipynb file to test the model on a sample input
---

## **Expected Output**
Upon running the demo.ipynb script, the predictions will be saved to the `results/` folder. Example:
- **Input**: Retinal fundus image from `demo/data/`.
- **Output**: txt file with predicted class labels and confidence scores.

For test dataset evaluation:
- Overall Accuracy: **87.22%**
- Per-Class Accuracy:
  - **Dry AMD**: 100%
  - **Glaucoma**: 92.95%
  - **Normal Fundus**: 100%
  - **Mild DR**: 42.16%
  - **Moderate DR**: 86.57%
  - **Severe DR**: 85.05%
  - **Proliferative DR**: 78.02%
  - **Cataract**: 100%
  - **Hypertensive Retinopathy**: 88.30%
  - **Pathological Myopia**: 88.24%

---

## **Pre-trained Model**
Download the pre-trained model from this [link](https://drive.google.com/file/d/1-G-jaO900piAxxhWqXtMdDIxVaPYGAjn/view?usp=drive_link).

Place the model in the `checkpoints/` directory. Use the following snippet to load the model:
```python
model.load_state_dict(torch.load("checkpoints/efficientnet_b0_pretrained.pth"))
```

---

## **Acknowledgments**
- Dataset: [Retinal Fundus Images (Kaggle)](https://www.kaggle.com/datasets/kssanjaynithish03/retinal-fundus-images)
- Pretrained Weights: [EfficientNet-B0 Weights](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html)
- Loss Function: Cross entropy

Special thanks to the creators of the dataset and the PyTorch community for providing robust tools for deep learning.

---