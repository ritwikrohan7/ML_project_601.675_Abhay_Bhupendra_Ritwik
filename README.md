# ML_project_601.675_Abhay_Bhupendra_Ritwik

### Project Name: Transformer-based Surgical Tool Segmentation
Team Mentor: Sophia Sklaviadis

Team Members:

i) Abhay Kodapurath Ajay

ii) Bhupendara Mahar

iii) Ritwik Rohan


### Using the code:

- Clone this repository using these command in jupyter notebook:

git clone https://github.com/ritwikrohan7/ML_project_601.675_Abhay_Bhupendra_Ritwik.git
%cd ML_project_601.675_Abhay_Bhupendra_Ritwik


The code is stable using Python 3.6.10, Pytorch 1.4.0


To install all the dependencies using pip, write the following command:


!pip install -r requirements.txt

### Codes in this git repo

i) metrics.py : This code has the loss function and the evaluation metrics functions in it.

ii) preprocess.py: This code can be used if you want to preprocess certain data.

iii) train.py: This code is used to train the model and the model can be selected as per the user's choice. The weighted model is saved in the google drive (link attached below this section).

iv) test.py: This code is used to test the trained model using the trained model.

v) utils.py and utils_gray.py: This code has all the preprocessing functions like jointtransform2D, imagetoimage2D etc.

## Using the Code for your dataset

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
python train.py --train_dataset "enter train directory" --val_dataset "enter validation directory" --direc 'path for results to be saved' --batch_size 4 --epoch 400 --save_freq 10 --modelname "gatedaxialunet" --learning_rate 0.001 --imgsize 128 --gray "no"
```

```bash
Change modelname to MedT or logo to train them
```

### Testing Command:

```bash 
python test.py --loaddirec "./saved_model_path/model_name.pth" --val_dataset "test dataset directory" --direc 'path for results to be saved' --batch_size 1 --modelname "gatedaxialunet" --imgsize 128 --gray "no"
```

