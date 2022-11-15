#%%
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
    # Output: {'danger': 0, 'safe': 1}

    # Getting the ordered list of filenames for the images
    image_files = pd.Series(image_generator.filenames)
    # image_files = list(image_files.str.split("\\", expand=True)[1].str[:-4])
    # image_generator.batch_size=image_files.shape[0]
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
def create_cnn(width, height, depth, filters=(16, 32, 64), regularizer=None):
    """
    Creates a CNN with the given input dimension and filter numbers.
    """
    # Initialize the input shape and channel dimension, where the number of channels is the last dimension
    inputShape = (height, width, depth)
    chanDim = -1
 
    # Define the model input
    inputs = Input(shape=inputShape)
 
    # Loop over the number of filters 
    for (i, f) in enumerate(filters):
        # If this is the first CONV layer then set the input appropriately
        if i == 0:
            x = inputs
 
        # Create loops of CONV => RELU => BN => POOL layers
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
    # Final layers - flatten the volume, then Fully-Connected => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(512, kernel_regularizer=regularizer)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)
 
    # Apply another fully-connected layer, this one to match the number of nodes coming out of the MLP
    x = Dense(256, kernel_regularizer=regularizer)(x)
    x = Activation("relu")(x)

    x = Dense(10, kernel_regularizer=regularizer)(x)
    x = Activation("relu")(x)
 
    # Construct the CNN
    model = Model(inputs, x)
 
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
width=180
height=400
depth=3
val_split=0.3
batch_size=32
image_type='cropped'
csv_file_name='all_pose_dataframe_{}_class_new.csv'.format(image_type)
image_folder = os.path.join(os.path.abspath(os.path.join(__file__,"../../..")),'data','{}_classes'.format(image_type))
model_name="Hybrid_Pose_Model_{}_class_new".format(image_type)
#%%
csv_files_path=os.path.join(os.path.abspath(os.path.join(__file__,"../../..")),'results')
csv_file=os.path.join(csv_files_path,csv_file_name)
model_path = os.path.join(os.path.abspath(os.path.join(__file__,"../..")),'models')    # path to save the trained model
df_original=pd.read_csv(csv_file,index_col=0)
df,image_gen=sort_index(df_original,image_folder,(width,height,depth))
# df.dropna(axis=0, inplace=True)
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
cnn = create_cnn(width, height, depth)

# Create the input to the final set of layers as the output of both the MLP and CNN
combinedInput = concatenate([mlp.output, cnn.output])


x = Dense(10, activation="relu")(combinedInput)
x= Dropout(0.5)(x)
x=Dense(4,activation='softmax')(x)
model = Model(inputs=[mlp.input,cnn.input], outputs=x)
model.summary()
#%%
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="sparse_categorical_crossentropy", metrics=['acc'], optimizer=opt)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6)
history =model.fit(x=[data_train,image_train],y=y_train,epochs=100, validation_data=([data_test,image_test],y_test),callbacks=[callback])

#%% Plots
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#%%
if not os.path.exists(model_path):
    os.makedirs(model_path)
model.save(os.path.join(model_path,model_name))
print('Model saved in ',model_path)
