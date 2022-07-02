import os
import pandas as pd
from tensorflow import keras
from keras import models
import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import cv2
from time import time


def load_image(img_path, target_shape):
    # load and scale the image
    img = keras.preprocessing.image.load_img(img_path, target_size=target_shape)
    img = keras.preprocessing.image.img_to_array(img)
    img = img.astype('float32')/255.0
    return img

def get_feature(img, encoder):
    img = np.expand_dims(img,0)
    feature = encoder.predict(img, verbose=0)
    return feature.flatten()


def get_features(img_paths,encoder):
    imgs = []
    for img_path in img_paths:
        img = keras.preprocessing.image.load_img(img_path, target_size=(224,224))
        img = keras.preprocessing.image.img_to_array(img)
        img = img.astype('float32')/255.0
        imgs.append(img)

    feature = encoder.predict(np.array(imgs), verbose=0)
    return feature

def save_all_labels(database_path, model_path, sace_path):
    model_comp = models.load_model(model_path)
    
    encoder = models.Sequential()
    encoder.add(models.Model(model_comp.input, model_comp.layers[-2].output))
    encoder.add(keras.layers.GlobalAveragePooling2D())
    encoder.trainable = False

    classifier = models.Model(model_comp.layers[-1].layers[1].input,model_comp.layers[-1].output)
    classifier.trainable = False

    database = pd.read_csv(database_path)

    batch = 128
    y_pred = []
    n=len(database)
    for i in range(0,n,batch):
        y_pred = y_pred + list(np.argmax(classifier.predict(get_features(database['image_name'].iloc[i:min(i+batch,n)],encoder),verbose=0),axis=1).flatten())

    label_map = [0,1,10,2,3,4,5,6,7,8,9]
    for i in range(n):
        y_pred[i] = label_map[y_pred[i]]
    
    database.insert(3, 'category_label', y_pred)
    database.to_csv(sace_path, index=False)

def save_all_features(databse_path, model_path, save_path):
    model_comp = models.load_model(model_path)
    
    encoder = models.Sequential()
    encoder.add(models.Model(model_comp.input, model_comp.layers[-2].output))
    encoder.add(keras.layers.GlobalAveragePooling2D())
    encoder.trainable = False

    database = pd.read_csv(databse_path)

    features = []
    batch = 128
    n = len(database)

    # compute the features of all images in the database.
    for i  in range(0,n,batch):
        features = features + get_features(database['image_name'].iloc[i:min(i+batch,n)],encoder)

    features_array = np.asarray(features)
    
    np.savez_compressed(save_path, arr_0 = features_array)


class recommender(object):
    def __init__(self, database_path, feature_path, model_path, label_map_path):
        # load the label_map
        self.label_map = []
        with open(label_map_path) as f:
            self.label_map = f.read().splitlines()
            
        # load the features of all images of database
        f = np.load(feature_path)
        self.all_features = f['arr_0']

        # load the database
        self.database = pd.read_csv(database_path)
        self.database.drop(['Unnamed: 0', 'feature'], axis=1, inplace=True)
        self.category_groups = self.database.groupby(self.database['category_label'])

        model_comp = models.load_model(model_path)
        self.encoder = models.Sequential()
        self.encoder.add(models.Model(model_comp.input, model_comp.layers[-2].output))
        self.encoder.add(keras.layers.GlobalAveragePooling2D())
        self.encoder.trainable = False

        self.classifier = models.Model(model_comp.layers[-1].layers[1].input,model_comp.layers[-1].output)
        self.classifier.trainable = False

        self.query_image = None
        self.neighbor_images = []
        self.label = None
        self.similarity = []

        print("Succesfully Initialized recommender object with", len(self.database), "images among", len(self.category_groups), "categories")
        pass

    def __load_image(self, img_path):
        return load_image(img_path, target_shape=self.encoder.input_shape[1:3])

    def __get_feature(self, img):
        return get_feature(img,self.encoder)
    
    def __get_label(self, feature, label_map = [0,1,10,2,3,4,5,6,7,8,9]):
        label = np.argmax(self.classifier.predict(np.expand_dims(feature,0), verbose=0))
        return label_map[label]
    
    def __get_neighbors(self,query, num_neighbors=9):
        #  compute neighbors of query in the database
        label = self.__get_label(query)
        all_samples = self.category_groups.get_group(label)
        neighs = [] # maintain top num_neighbors

        for i in all_samples.index:
            feat = self.all_features[i]
            sim = 1-cosine(query,feat)
        
            if(sim==1):
                continue
        
            neighs.append((sim,i))
            neighs.sort(key=lambda x:x[0], reverse=True)
        
            if(len(neighs)>num_neighbors):
                neighs.pop(-1)

        dists = [i for i,j in neighs]
        neighs = [j for i,j in neighs]
        return neighs,dists, label
    
    def __random_choose(self):
        # randomly choose a query image from the database
        num_class = len(self.category_groups)
        label = np.random.choice(num_class)
        all_samples = self.category_groups.get_group(label)
        query_index = np.random.choice(all_samples.index)
        return self.database['image_name'][query_index], query_index
    
    def get_label(self, img_path, label_map = [0,1,10,2,3,4,5,6,7,8,9]):
        img = self.__load_image(img_path)
        feat = self.__get_feature(img)
        return self.__get_label(feat,label_map)
    
    def show(self):
        query_image = self.query_image
        neighbors = self.neighbor_images
    
        # show the query image and the neighbors
        if(len(neighbors)==0):
            print("No neighbors found! first call get_similar() to get similar images")
            return
        
        n = int(np.sqrt(len(neighbors)))
        
        plt.figure(figsize=(5,5))
        plt.imshow(query_image)
        plt.axis('off')
        plt.title('Query Image' + '\n' + 'Predicted Category: ' + self.label_map[self.label])
        
        plt.figure(figsize=(5*n,5*n))
        for i in range(n*n):
            plt.subplot(n,n,i+1)
            plt.imshow(neighbors[i])
            plt.axis('off')
            plt.title(f"Simmilarity: {self.similarity[i]*100:.2f} %")
        plt.show()
    
    def get_similar(self, query_image_path = None, num_neighbors=9):
        # get the similar recommendations for query image
        start = time()
        query_feat = []
        if query_image_path is None:
            query_image_path, ind = self.__random_choose()
            query_feat = self.all_features[ind]
        
        self.query_image = self.__load_image(query_image_path)
        
        if len(query_feat)==0:
            query_feat = self.__get_feature(self.query_image)

        neighs,dists, label = self.__get_neighbors(query_feat, num_neighbors)

        neighbor_images = []
        for i in neighs:
            neighbor_images.append( self.__load_image(self.database['image_name'][i]) ) 
        
        self.neighbor_images = neighbor_images
        self.label = label
        self.similarity = dists

        print(f"Found {len(neighbor_images)} similar images in {time()-start:.2f} seconds")
        return self.query_image, neighbor_images, label

class counter:
    def __init__(self,model_path, label_path):
        model_comp = models.load_model(model_path)
        self.encoder = models.Sequential()
        self.encoder.add(models.Model(model_comp.input, model_comp.layers[-2].output))
        self.encoder.add(keras.layers.GlobalAveragePooling2D())
        self.encoder.trainable = False

        self.classifier = models.Model(model_comp.layers[-1].layers[1].input,model_comp.layers[-1].output)
        self.classifier.trainable = False

        self.label_map = []
        with open(label_path) as f:
            self.label_map = f.read().splitlines()
        self.labels = None
        self.imgs = []
        print("Succesfully Initialized counter object")

    def __get_label(self, img_path, label_map = [0,1,10,2,3,4,5,6,7,8,9]):
        img = load_image(img_path, target_shape=self.encoder.input_shape[1:3])
        feat = get_feature(img,self.encoder)
        label = np.argmax(self.classifier.predict(np.expand_dims(feat,0), verbose=0))
        (self.imgs).append(img)
        return label_map[label]

    def __get_label_batch(self, img_paths):
        labels = []
        for img_path in img_paths:
            labels.append(self.__get_label(img_path))
        return labels

    def __get_top_klabels(self, k):
        labels = self.labels
        freq = [ (i,labels.count(i)) for i in set(labels) ]
        freq.sort(key=lambda x: x[1], reverse=True)
        return freq[:min(k,len(freq))]
    
    def show_imgs(self):
        plt.figure(figsize=(5*len(self.imgs),5))
        for i, img in enumerate(self.imgs):
            plt.subplot(1,len(self.imgs),i+1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Predicted Category: {self.label_map[self.labels[i]]}")
        plt.show()


    def count(self, dir_path=None, img_paths=[], k=None):
        self.labels = []
        self.imgs = []

        if k is None:
            k = len(self.label_map)
        
        if dir_path is not None:
            img_paths=[]
            for i in os.listdir(dir_path):
                img_paths.append(os.path.join(dir_path,i))
        
        self.labels = self.__get_label_batch(img_paths)
        toplabels = self.__get_top_klabels(k)
        
        print(f"Top {len(toplabels)} labels:")
        print("Lable\tCount")
        for i in toplabels:
            print(f"{self.label_map[i[0]]}: {i[1]}")
    