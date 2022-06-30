# Fashion Recommendation System
This system could show similar garments from a fixed database.

I trained a model to classify clothes into 11 catefories which are as follows-
- Blazer
- Cardigan
- Hoodie
- Jacket
- Sweater
- Tank
- Tee
- Top
- Dress
- Jumpsuit
- Romper

## Dataset
I used a subset of <a href = "https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html" > deep fashion dataset</a>. <br>
This dataset has 50 types of clothing, but quality of labels is challenging due to fact that it contains many images showing both upper and lower body cloths but belonging to only one category. Due to the time and resoure constrains I could not dive deeper into it, so I **used only upper/full** body clothes. <br>
To make the training set balanced I further rejected those categories that had less than 2000 samples, and for those categories which had more than 7000 samples I took random 7000 samples, though to create the database i took all the samples.

This way I created a subset for training containing above 11 categories and each category has **number of samples between 2500 and 7000**. <br>
The size of training set was ~65K images which was further divided into training and validation set in the **ratio 80:20**. <br>
The complete database (from which similar cloths would be drawn) contains around **140K images**.

You could view other stats and some sample images in the <a href= "data_exploration.ipynb" > ./data_exploration.ipynb </a> notebook.
 
## Model
I experimented with many base models such as ResNet50, InceptionV3, EfficientNet, DenseNet169 and Xception. <br>
While **InceptionV3** performed good **(60% accuracy)**, <br>
**Xception** model gave best results, with **~65%** validation accuracy, and **~70**% training accuracy, though the number are small, the model generalized very nicely and gave **~75%** accuracy on the complete database! (this is mainly due to the fact that the database is biased towards some categories). I tested the model on some random ~20 images taken from google and it predicted all of them correctly!

Due to time constrains I did'not tweaked much with the architechture, but the results shows that the model could have gotten much higher accuraries.

A small fully connected neural network with **3 Dense** layers was used as classification head.

## Recommendation System
Once the model was trained, I computed the features (2048 dimensional) of all the images in the datbase and stored them as a matrix in a .npy file.

To find the images similar to a give image (query), the following algorithm is used-

- Compute the features and predict the label of the query image.
- Compute the **cosine similarity** of these features with the features of all those images from the datbase which have the same label
- Maintain the top n images whose similarity is maximum.

## Finding Top-3 garments from a set of images
This was much simpler, All I had to was to predict the labels of all the images int he set and done!

**THE NOTEBOOK <a href="./results.ipynb"> ./results.ipynb </a>  SHOWS SOME RESULTS OBTAINED.**