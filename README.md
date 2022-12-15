# ML_project_601.675_Abhay_Bhupendra_Ritwik
# Project Name: Transformer-based Surgical Tool Segmentation
Team Mentor: Sophia Sklaviadis

Team Members:
i) Abhay Kodapurath Ajay (fkodapu1@jh.edu)
ii) Bhupendra Mahar (bmahar1@jh.edu)
iii) Ritwik Rohan (rrohan2@jh.edu)

### Our Dataset
We have kept our dataset in this google drive. Also you need to mount this drive in the notebook to run the codes.

Google drive link: https://drive.google.com/drive/folders/1Gz29NftwsLkPvRwlYkzjzYFO1qsGCdtH

### Codes in the git repo
i) metrics.py - Code for loss functions and F1 and IoU scores

ii) preprocess.py - Code if you want to run just the preproceesing of data.

iii) train.py - Code for training the model according to the model name given in notebook command

iv) test.py - Code for testing the repo and saving the final weights to use in metrics.py

v) utils.py and utils_gray.py - Code for the preprocessing of data which is used in train.py and test.py 


### Using the code in jupyter notebook:

- Clone this repository:
```bash
!git clone https://github.com/ritwikrohan7/ML_project_601.675_Abhay_Bhupendra_Ritwik.git
%cd ML_project_601.675_Abhay_Bhupendra_Ritwik
```

The code is stable using Python 3.6.10, Pytorch 1.4.0

To install all the dependencies using pip:

```bash
pip install -r requirements.txt
```
## Using the Code for any dataset

### Dataset Preparation

Prepare the dataset in the following format for easy use of the code. The train and test folders should contain two subfolders each: img and label. Make sure the images their corresponding segmentation masks are placed under these folders and have the same name for easy correspondance. Please change the data loaders to your need if you prefer not preparing the dataset in this format.



```bash
Train Folder-----
      img----
          0001.png
          0002.png
          .......
      labelcol---
          0001.png
          0002.png
          .......
Validation Folder-----
      img----
          0001.png
          0002.png
          .......
      labelcol---
          0001.png
          0002.png
          .......
Test Folder-----
      img----
          0001.png
          0002.png
          .......
      labelcol---
          0001.png
          0002.png
          .......

```

- The ground truth images should have pixels corresponding to the labels. Example: In case of binary segmentation, the pixels in the GT should be 0 or 255.

### Training Command:

```bash 
!python train.py --train_dataset "/content/drive/MyDrive/MLProject/dataset/Train_resized" --val_dataset "/content/drive/MyDrive/MLProject/dataset/Validation_resized" --direc '/content/drive/MyDrive/MLProject/dataset/Results' --batch_size 4 --epoch 400 --save_freq 10 --modelname "MedT" --learning_rate 0.001 --imgsize 450 --gray "no"
```

```bash
Change modelname to MedT or logo to train them according to the type of model you prefer.
```

### Testing Command:

```bash 
python test.py --loaddirec "./saved_model_path/model_name.pth" --val_dataset "test dataset directory" --direc 'path for results to be saved' --batch_size 1 --modelname "gatedaxialunet" --imgsize 128 --gray "no"
```
