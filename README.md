# Recognizing numbers in natural scene images using MSER and CNN

Program will take input images from input_images folder and save output image into graded_images. 
Program will printout digits in left to right, top to bottom order.

Program was developed using Python 3. The CNN model svhn_cnn_format1.pth was trained on Google Colab. Program with trained model were tested in cv_proj.yml enviroment and MacOS 10.14 without GPU.

Please note that if the trained model svhn_cnn_format1.pth is deleted, the program will download, propesse and train the CNN model from scratch. This function is not fully tested under local enviroment. 

Image example, house numbers sign recognized using this progrom

![Output example](/graded_images/4.png)
