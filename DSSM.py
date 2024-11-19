import tensorflow as tf 
import numpy as np  
import pandas as pd 
from sklearn.model_selection import train_test_split
import os
inputs = {
    'user_id':tf.keras.Input(name = 'user_id',shape = (),dtype = 'int32'),
    'item_id':tf.keras.Input(name = 'item_id',shape = (),dtype = 'int32'),
    'price':tf.keras.Input(name = 'price',shape = (),dtype = 'float32'),
    'user_age':tf.keras.Input(name = 'user_age',shape = (),dtype = 'float32')
}
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


import numpy as np
import tensorflow as tf

class UserTower(tf.keras.Model):
    def __init__(self):
        super().__init__()  
        self.user_age_bucket_layer = tf.keras.layers.Discretization(bin_boundaries=np.arange(1, 100, 10).tolist())
        self.user_id_embedding_layer = tf.keras.layers.Embedding(input_dim=101, output_dim=32)
        self.user_age_embedding_layer = tf.keras.layers.Embedding(input_dim=11, output_dim=32)
        self.first_batchnorm_layer = tf.keras.layers.BatchNormalization()
        self.first_dense_layer = tf.keras.layers.Dense(32, activation='relu')
        self.second_batchnorm_layer = tf.keras.layers.BatchNormalization()
        self.second_dense_layer = tf.keras.layers.Dense(16, activation='relu')

    @tf.function
    def call(self, inputs):
        user_age_bucket = self.user_age_bucket_layer(inputs['user_age'])
        user_age_embedding = self.user_age_embedding_layer(user_age_bucket)
        user_id_embedding = self.user_id_embedding_layer(inputs['user_id'])
        features = [user_id_embedding, user_age_embedding]
        features = tf.concat(features, axis=-1)
        dense1 = self.first_batchnorm_layer(features)
        dense1 = self.first_dense_layer(dense1)
        dense2 = self.second_batchnorm_layer(dense1)
        dense2 = self.second_dense_layer(dense2)
        return dense2

class ItemTower(tf.keras.Model):
    def __init__(self):
        super().__init__()  
        self.item_id_embedding = tf.keras.layers.Embedding(input_dim=101, output_dim=32)
        self.price_bucket_layer = tf.keras.layers.Discretization(bin_boundaries=np.arange(1, 100, 10).tolist())
        self.price_embedding_layer = tf.keras.layers.Embedding(input_dim=11, output_dim=32)
        self.first_batchnorm_layer = tf.keras.layers.BatchNormalization()
        self.first_dense_layer = tf.keras.layers.Dense(32, activation='relu')
        self.second_batchnorm_layer = tf.keras.layers.BatchNormalization()
        self.second_dense_layer = tf.keras.layers.Dense(16, activation='relu')

    @tf.function
    def call(self, inputs):
        item_price_bucket = self.price_bucket_layer(inputs['price'])
        item_price_embedding = self.price_embedding_layer(item_price_bucket)
        item_embedding = self.item_id_embedding(inputs['item_id']) 
        features = [item_price_embedding, item_embedding]
        features = tf.concat(features, axis=-1)
        dense1 = self.first_batchnorm_layer(features)
        dense1 = self.first_dense_layer(dense1)
        dense2 = self.second_batchnorm_layer(dense1)
        dense2 = self.second_dense_layer(dense2)
        return dense2

class DSSM(tf.keras.Model):
    def __init__(self):
        super().__init__()  
        self.user_tower = UserTower()
        self.item_tower = ItemTower()


    @tf.function
    def call(self, inputs):
        item_dense = self.item_tower(inputs)
        user_dense = self.user_tower(inputs)
        pred = tf.reduce_sum(tf.multiply(item_dense, user_dense),axis=1,keepdims=True)
        prob = tf.keras.activations.sigmoid(pred)
        return prob
model = DSSM()
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn =tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer=opt,
                 loss=loss_fn,
                 metrics=[tf.keras.metrics.AUC()])

model.fit(df_to_dataset(train),validation_data = df_to_dataset(test),epochs= 4)
print(model)
