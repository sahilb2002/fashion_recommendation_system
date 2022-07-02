# Fashion Recommendation System
View on github- https://github.com/sahilb2002/fashion_recommendation_system 

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

## Tesing the code
To test the code follow these instructions-
- Download the model weights from <a href = "https://drive.google.com/drive/folders/1VEK2PZYhm34633uRfaj-teToRwSdM6dK?usp=sharing"> here </a> (download the complete folder).

- If you have no other database, download the database from <a href="https://drive.google.com/file/d/1-kxFk9ojVXV2Nx2vsitmX-3F7suQFwCm/view?usp=sharing">here</a>, (download the tar file and extract the dataset folder from it).

- Prepare a csv file as follows- it should contain two columns named **image_name** and **category_label**, the image_name columns should contains paths of images relative to CWD, and category_label should contain the category labels from [0,10] as per the above list of clothes, if you dont have the labels compute labels as follows-
```
from fashion_studio import save_all_labels
save_all_labels(database_path = /path/to/csv/containing_image_paths_as_described, model_path=/path/to/model, save_path=/path/to/save_csv )
```
**NOTE** If you are using the above database, you can use the <a href="./features.csv">./features.csv</a> file as your database file, but ensure that the extracted dataset folder is in **parent directory of CWD**.

- Computing features-
```
from fashion_studio import save_all_features
save_all_features(database_path = /path/to/csv/containing_2_columns_as_described, model_path=/path/to/model, save_path = /path/to/save/.npz_file)
```
Alternatively you can view the features.ipynb file.

**NOTE** Again if you are using the above database you can download this .npz file from <a href="https://drive.google.com/file/d/1-2WBZyjwHcQFrffnIEQDU8UR47psgGpB/view?usp=sharing">here</a> (around 900 MB)

- Download label map from <a href="https://drive.google.com/file/d/1LmF9oxGa8SUF8xI5zqlMqFgkhGH98CXG/view?usp=sharing"> here </a>

- Finally get similar recommendation using below code-
```
from fashion_studio import recommender
recommend = recommender(database_path = /path/to/csv, feature_path = /path/to/.npz_file, model_path=/path/to/model, label_map_path = /path/to/txt)
```
If you have a test image-
```
recommend.get_similar(/path/to/test_image, num_neighbors = number_of_recommendation_you_want(default 9)) # returns query_image, list of similar_images and label_of_image
recommend.show()
```
Else it will take a random image from database-
```
recommend.get_similar()
recommend.show()
```

- To identify most common category from a set of images-
Put all the images in a directory, or you need a list of paths_to_all_images
```
from fashion_studio import counter
count = counter(model_path = /path/to/model, label_path = /path/to/label_map.txt)

count.count(dir_path = /path/to/dir_containing_images or img_paths = [list containing image_paths], k = number_of_labels_to_be_shown) # at least one of dir_path, img_paths must be specified, if both are specified dir_path is used.

count.show() # to see all images and their predicted category.
```