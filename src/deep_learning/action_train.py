#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, Input, concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
import os
#%%
def create_mlp(dim, regularizer=None):
    """Creates a simple two-layer MLP with inputs of the given dimension"""
    model = Sequential()
    model.add(Dense(20, input_dim=dim, activation="relu", kernel_regularizer=regularizer))
    model.add(Dense(10, activation="relu", kernel_regularizer=regularizer))
    model.add(Dense(10, activation="relu", kernel_regularizer=regularizer))
    return model

#%%
val_split=0.3
csv_file_name='all_pose_dataframe.csv'
csv_files_path=os.path.join(os.path.abspath(os.path.join(__file__,"../../..")),'results')
csv_file=os.path.join(csv_files_path,csv_file_name)
model_path = os.path.join(os.path.abspath(os.path.join(__file__,"../..")),'models')    # path to save the trained model
model_name="MLP_Pose_Model"
df=pd.read_csv(csv_file,index_col=0)
df.dropna(axis=0, inplace=True)
print(df['label'].unique())
labels=df['label'].str.split("_").str[0].astype(int)
data=df.drop(columns='label').values
print(labels)
print(labels.dtype)
x_train, x_test, y_train, y_test = train_test_split(data, labels,test_size=val_split)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
mlp = create_mlp(x_train.shape[1])
x = Dense(4, activation="relu")(mlp.output)
x = Dense(4, activation="sigmoid")(x)
model = Model(inputs=mlp.input, outputs=x)
model.summary()
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="sparse_categorical_crossentropy", metrics=['acc'], optimizer=opt)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6)
history =model.fit(x_train,y_train,epochs=500, callbacks=[callback],validation_data=(x_test, y_test))


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
