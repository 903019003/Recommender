
import tensorflow as tf
import numpy as np  
import pandas as pd 
from sklearn.model_selection import train_test_split
import os

data_dict = {'user_id':np.random.randint(1000,1100,size = 100),
             'user_age':np.random.randint(1,99,size = 100),
             'item_id':np.random.randint(1000,1100,size =100),
             'price':np.random.uniform(1000,1100,size =100),
             'label':np.random.randint(0,2,size = 100)}
df = pd.DataFrame(data_dict)
df['user_age'] = df['user_age'].astype('float32')
df['price'] = df['price'].astype('float32')
df['user_id'] = df['user_id'].astype('int32')
df['item_id'] = df['item_id'].astype('int32')
print(df.dtypes)
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('label')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds
train, test = train_test_split(df, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

df_to_dataset(train)

class widedeep(tf.keras.Model):
    def __init__(self):
        super().__init__(widedeep)
        ## embedding_layer
        self.user_id_embedding_layer = tf.keras.layers.Embedding(input_dim=101, output_dim=32,name = "user_id_embedding")
        self.item_id_embedding_layer = tf.keras.layers.Embedding(input_dim=101, output_dim=32,name = "item_id_embedding")
        self.user_age_embedding_layer = tf.keras.layers.Embedding(input_dim = 11, output_dim=8,name = "user_age_embedding")
        self.price_embedding_layer = tf.keras.layers.Embedding(input_dim = 11,output_dim = 8)
        ## bucket_layer
        self.user_age_bucket_layer = tf.keras.layers.Discretization(bin_boundaries=np.arange(1,100,10).tolist())
        self.price_bucket_layer = tf.keras.layers.Discretization(bin_boundaries=np.arange(1000,1100,10).tolist())
        ## network
        self.first_batchnorm_layer  = tf.keras.layers.BatchNormalization()
        self.first_dense_layer = tf.keras.layers.Dense(128,activation='relu')
        self.second_batchnorm_layer = tf.keras.layers.BatchNormalization()
        self.second_dense_layer = tf.keras.layers.Dense(64,activation='relu')
        self.third_batchnorm_layer = tf.keras.layers.BatchNormalization()
        self.output_layer = tf.keras.layers.Dense(1, kernel_regularizer='l2',name="output_dense_layer")
    
    def build_model(self,wide_feature,deep_feature):
        hidden1 = self.first_batchnorm_layer(deep_feature)
        hidden1 = self.first_dense_layer(hidden1)
        hidden2 = self.second_batchnorm_layer(hidden1)
        hidden2 = self.second_dense_layer(hidden2)
        hidden3 = self.third_batchnorm_layer(hidden2)
        features = tf.keras.layers.concatenate([hidden3,wide_feature],axis = -1)
        pred = self.output_layer(features)
        prob = tf.keras.activations.sigmoid(pred)
        return prob 
    @tf.function
    def call(self,inputs):
        user_age_bin = self.user_age_bucket_layer(inputs['user_age'])
        user_age_ohe = tf.one_hot(user_age_bin,depth = 10)
        price_bin = self.price_bucket_layer(inputs['price'])
        price_ohe = tf.one_hot(price_bin,depth = 10)
        ## embedding
        user_id_embedding = self.user_id_embedding_layer(inputs['user_id'])
        item_id_embedding = self.item_id_embedding_layer(inputs['item_id'])
        price_embedding = self.price_embedding_layer(price_bin)
        age_embedding = self.user_age_embedding_layer(user_age_bin)
        ## cross 
        user_X_item = user_id_embedding * item_id_embedding
        age_X_price = price_embedding * age_embedding
        wide_feature = [user_age_ohe,price_ohe]
        deep_feature = [user_id_embedding,item_id_embedding,price_embedding,age_embedding,age_X_price,user_X_item]
        wide_feature = tf.keras.layers.Concatenate(axis=-1)(wide_feature)
        deep_feature = tf.keras.layers.Concatenate(axis=-1)(deep_feature)
        return self.build_model(wide_feature,deep_feature)

model = widedeep()
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn =tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer=opt,
                 loss=loss_fn,
                 metrics=[tf.keras.metrics.AUC()])

model.fit(df_to_dataset(train),validation_data = df_to_dataset(test),epochs= 4)
