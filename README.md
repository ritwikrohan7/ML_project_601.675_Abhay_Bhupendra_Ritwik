# ML_project_601.675_Abhay_Bhupendra_Ritwik

### Project Name: Transformer-based Surgical Tool Segmentation
Team Mentor: Sophia Sklaviadis

Team Members:

i) Abhay Kodapurath Ajay (fkodapu1@jh.edu)

ii) Bhupendara Mahar (bmahar1@jh.edu)

iii) Ritwik Rohan (rrohan2@jh.edu)


### Using the code:

Mount the google drive in the jupyter notebook using this command:
```bash
from google.colab import drive
drive.mount('/content/drive')
```

- Clone this repository using these command in jupyter notebook:

```bash
git clone https://github.com/ritwikrohan7/ML_project_601.675_Abhay_Bhupendra_Ritwik.git
%cd ML_project_601.675_Abhay_Bhupendra_Ritwik
```


The code is stable using Python 3.6.10, Pytorch 1.4.0


To install all the dependencies using pip, write the following command:


!pip install -r requirements.txt

### Codes in this git repo

i) metrics.py : This code has the loss function and the evaluation metrics functions in it.

ii) preprocess.py: This code can be used if you want to preprocess certain data.

iii) train.py: This code is used to train the model and the model can be selected as per the user's choice. The weighted model is saved in the google drive (link attached below this section).

iv) test.py: This code is used to test the trained model using the trained model.

v) utils.py and utils_gray.py: This code has all the preprocessing functions like jointtransform2D, imagetoimage2D etc.

vi) lib/model/model_codes.py: This code has the forward function for all the models that we used to compare our proposed model with.


You can access the datasets from the google drive link. We have given the drive access to anyone with this link. Google drive link: https://drive.google.com/drive/folders/1Gz29NftwsLkPvRwlYkzjzYFO1qsGCdtH?usp=sharing


## Using the Code for your dataset

### Dataset Preparation

The datasets are already present in the google drive (link above). In case you need to run with different datasets, follow the procedure below:

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
### Before Training:

The default size for image is 128. If you want to change the image dimension, you have to change the dimension in the 4 resizing cells. The command looks like this: "new_image = cv2.resize(a,(128,128))". Change the value in all 4 cells if you want to chnage the dimension. Also change image size in the train command below


### Training Command in jupyter notebook:

```bash 
!python train.py --train_dataset "/content/drive/MyDrive/MLProject/dataset/Train_resized" --val_dataset "/content/drive/MyDrive/MLProject/dataset/Validation_resized" --direc '/content/drive/MyDrive/MLProject/dataset/Results' --batch_size 4 --epoch 400 --save_freq 10 --modelname "MedT" --learning_rate 0.001 --imgsize 128 --gray "no"
```

```bash
Change modelname to MedT (our proposed model) or logo (only local-global training) to train them according to the model required. 
```

### Testing Command in jupyter notebook:

```bash 
!python test.py --train_dataset "/content/drive/MyDrive/MLProject/dataset/Train_resized" --loaddirec "/content/drive/MyDrive/MLProject/dataset/Results/390/MedT.pth" --val_dataset "/content/drive/MyDrive/MLProject/dataset/Train_resized" --direc '/content/drive/MyDrive/MLProject/test_set/Results/' --batch_size 1 --modelname "MedT" --imgsize 128 --gray "no"
```

