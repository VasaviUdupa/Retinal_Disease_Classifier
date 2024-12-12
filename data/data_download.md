### Instructions to Download and Organize Data

#### 1. **Download the Dataset**
   - The dataset for Retinal Fundus Image Classification is available on [Kaggle](https://www.kaggle.com/datasets/kssanjaynithish03/retinal-fundus-images/data).
   - You can either:
     - Download it manually by clicking the download button on Kaggle.
     - Use the Kaggle API to download programmatically.

#### 2. **Using the Kaggle API**
   - Install the Kaggle API:
     ```bash
     pip install kaggle
     ```
   - Set up the Kaggle API by placing your Kaggle API key (`kaggle.json`) in the directory `~/.kaggle/`. You can download the API key from your Kaggle account under "Account Settings."
   - Run the following command to download the dataset:
     ```bash
     kaggle datasets download -d kssanjaynithish03/retinal-fundus-images
     ```
   - Unzip the downloaded file:
     ```bash
     unzip retinal-fundus-images.zip -d data/
     ```

#### 3. **Organize the Data**
   - Place the unzipped dataset in the `data/` folder of your project directory.
   - Your project folder structure should look like this:
     ```
     ├── data/
     │   ├── Retinal Fundus Images/
     │       ├── train/
     │       ├── val/
     │       ├── test/
     ├── src/
     │   ├── main.py
     │   ├── utils.py
     │   ├── model.py
     ├── requirements.txt
     └── README.md
     ```

#### 4. **Verify Dataset**
   - Ensure the dataset paths in your code match the directory structure.
   - Example dataset paths:
     - Training data: `data/Retinal Fundus Images/train/`
     - Validation data: `data/Retinal Fundus Images/val/`
     - Test data: `data/Retinal Fundus Images/test/` 

#### 5. **Note**
   - If using Google Colab, upload the dataset to your Google Drive and update paths in the code accordingly:
     ```python
     train_path = r"/content/drive/My Drive/RetinalFundusImages/Retinal Fundus Images/train"
     val_path = r"/content/drive/My Drive/RetinalFundusImages/Retinal Fundus Images/val"
     ```