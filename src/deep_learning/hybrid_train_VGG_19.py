#%%
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, Input, concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import tensorflow as tf
import itertools
import pandas as pd
import numpy as np
import os
#%%
def prepare_data(path,image_gen,img_height,img_width):
    # classes=os.listdir(path)
    image_data=[]
    for i in image_gen.filenames:
        img=img_to_array((Image.open(os.path.join(path,i))).resize((img_width,img_height)))
        image_data.append(img)
    image_data=np.array(image_data)
    # labels=np.array(image_gen.labels)
    return image_data
#%%
def sort_index(df,image_folder,dim):

    width,height,depth=dim
    
    df['file_names']=df.index

    image_generator = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
            dataframe=df,
            directory=image_folder,
            target_size=(height, width),
            x_col="file_names",
            y_col="label")

    # images, labels = next(image_generator)
    # Checking the labels
    image_generator.class_indices
    

    # Getting the ordered list of filenames for the images
    image_files = pd.Series(image_generator.filenames)

    # Sorting the structured data into the same order as the images
    df_sorted = df.reindex(image_files)
    df_sorted=df_sorted.dropna()
    df_sorted.drop(columns='file_names',inplace=True)
    return df_sorted,image_generator

#%%
def create_mlp(dim, regularizer=None):
    """Creates a simple two-layer MLP with inputs of the given dimension"""
    model = Sequential()
    model.add(Dense(20, input_dim=dim, activation="relu", kernel_regularizer=regularizer))
    model.add(Dense(10, activation="relu", kernel_regularizer=regularizer))
    model.add(Dense(10, activation="relu", kernel_regularizer=regularizer))
    return model

#%%
def create_cnn(width, height, depth, classes):
    #%% Defining the imagenet-trained VGG16 model

    # The model doesn't support GRAY image i.e. channels=1
    model=VGG19(weights='imagenet',include_top=False,input_shape=(height,width,3),classes=len(classes))
    for layer in model.layers:
        layer.trainable=False
    # Adding further layers in the pre-trained model
    x=Flatten()(model.output)
    x=Dense(units=512,activation='relu')(x)
    x= Dropout(0.5)(x)
    x=Dense(units=10,activation='relu')(x)
    # x= Dropout(0.5)(x)
    x=Dense(len(classes),activation='softmax')(x)
    model=Model(model.input,x)
 
    # Return the CNN
    return model

#%%
def data_generator(data,batch_size):
    it=iter(data)
    while True:
        chunk = tuple(itertools.islice(it, batch_size))
        chunk=np.array(chunk)
        # if pose_data.shape[0]==0:
        #     break
        yield chunk
#%%
def combined_generator(image_gen,pose_gen):
    while True:
        pose_data=next(pose_gen)
        images, labels = next(image_gen)
        yield [pose_data,images],labels

#%%
width=250		# standard width of the images for the trained model
height=550		# standard height of the images for the trained model
depth=3
val_split=0.3		# validation split for the training
batch_size=16
epochs=2		#total training epochs
csv_file_name='pose_filtered_talal_dataframe.csv'
model_name="Hybrid_VGG19_Model"
image_folder = os.path.join(os.path.abspath(os.path.join(__file__,"../../..")),'data','cropped_data')

#%%
csv_files_path=os.path.join(os.path.abspath(os.path.join(__file__,"../../..")),'results')
csv_file=os.path.join(csv_files_path,csv_file_name)
model_path = os.path.join(os.path.abspath(os.path.join(__file__,"../..")),'models')    # path to save the trained model
df_original=pd.read_csv(csv_file,index_col=0)
df,image_gen=sort_index(df_original,image_folder,(width,height,depth))
# df.dropna(axis=0, inplace=True)
df.to_csv(csv_file)
print('csv file written in ',csv_file)
classes=df['label'].unique()
classes.sort()
print(df['label'].unique())
labels=df['label'].str.split("_").str[0].astype(int).values
data=df.drop(columns='label').values

image_data=prepare_data(image_folder,image_gen,height,width)

# df=df.drop(columns='label')
# pose_gen=data_generator(data,batch_size)
# generator=combined_generator(image_gen,pose_gen)
print(labels)
data_train, data_test, image_train, image_test,y_train, y_test = train_test_split(data, image_data, labels,test_size=val_split)
#%%

mlp = create_mlp(data_train.shape[1])
cnn = create_cnn(width, height, depth,classes)

# Create the input to the final set of layers as the output of both the MLP and CNN
combinedInput = concatenate([mlp.output, cnn.output])


x = Dense(10, activation="relu")(combinedInput)
x= Dropout(0.5)(x)
x=Dense(len(classes),activation='softmax')(x)
model = Model(inputs=[mlp.input,cnn.input], outputs=x)
model.summary()
#%%
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="sparse_categorical_crossentropy", metrics=['acc'], optimizer=opt)

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', min_delta=0.05, verbose=2, patience=20),
    tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_path,'hybrid_vgg19_nov'), monitor='val_loss', mode='min', save_best_only=True, verbose=1),
    tf.keras.callbacks.TensorBoard(log_dir=os.path.join(model_path,'logs'))
]

history =model.fit(x=[data_train,image_train],y=y_train,batch_size=batch_size,epochs=epochs, validation_data=([data_test,image_test],y_test),callbacks=my_callbacks)

#%% Plots
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
#%%
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
if not os.path.exists(model_path):
    os.makedirs(model_path)
model.save(os.path.join(model_path,model_name))
print('Model saved in ',model_path)