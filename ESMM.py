
import tensorflow as tf
import numpy as np  
import pandas as pd 
from sklearn.model_selection import train_test_split
import os
data_dict = {'user_id':np.random.randint(1000,1100,size = 100),
             'user_age':np.random.randint(1,99,size = 100),
             'item_id':np.random.randint(1000,1100,size =100),
             'price':np.random.uniform(1000,1100,size =100),
             'label':np.random.randint(0,3,size = 100)}
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

class ESMM(tf.keras.Model):
    def __init__(self):
       super().__init__()

       ## bucket_layer
       self.user_age_bucket_layer = tf.keras.layers.Discretization(bin_boundaries=np.arange(1,100,10).tolist())
       self.price_bucket_layer = tf.keras.layers.Discretization(bin_boundaries=np.arange(1000,1100,10).tolist())
       ## embedding_layer 
       self.user_id_embedding_layer = tf.keras.layers.Embedding(input_dim = 101,output_dim = 32)
       self.item_id_embedding_layer = tf.keras.layers.Embedding(input_dim = 101,output_dim = 32)
       self.user_age_embedding_layer = tf.keras.layers.Embedding(input_dim = 11, output_dim=8,name = "user_age_embedding")
       self.price_embedding_layer = tf.keras.layers.Embedding(input_dim = 11,output_dim = 8)
       ##ctr_layer 
       self.ctr_batchnorm_layer1 = tf.keras.layers.BatchNormalization()
       self.ctr_dense_layer1 = tf.keras.layers.Dense(64,activation = 'relu')
       self.ctr_batchnorm_layer2 = tf.keras.layers.BatchNormalization()
       self.ctr_dense_layer2 = tf.keras.layers.Dense(32,activation = 'relu')
       self.ctr_batchnorm_layer3 = tf.keras.layers.BatchNormalization()
       self.ctr_logit_output_layer = tf.keras.layers.Dense(1,kernel_regularizer = 'l2')
       ##ctcvr_layer 
       self.ctcvr_batchnorm_layer1 = tf.keras.layers.BatchNormalization()
       self.ctcvr_dense_layer1 = tf.keras.layers.Dense(64,activation = 'relu')
       self.ctcvr_batchnorm_layer2 = tf.keras.layers.BatchNormalization()
       self.ctcvr_dense_layer2 = tf.keras.layers.Dense(32,activation = 'relu')
       self.ctcvr_batchnorm_layer3 = tf.keras.layers.BatchNormalization()
       self.ctcvr_logit_output_layer = tf.keras.layers.Dense(1,kernel_regularizer = 'l2')
       

    def build_ctr(self,features):
       dense1 = self.ctr_batchnorm_layer1(features)
       dense1 = self.ctr_dense_layer1(dense1)
       dense2 = self.ctr_batchnorm_layer2(dense1)
       dense2 = self.ctr_dense_layer2(dense2)
       dense3 = self.ctr_batchnorm_layer3(dense2)
       logit = self.ctr_logit_output_layer(dense3)
       return tf.keras.activations.sigmoid(logit)

    def build_ctcvr(self,features):
       dense1 = self.ctcvr_batchnorm_layer1(features)
       dense1 = self.ctcvr_dense_layer1(dense1)
       dense2 = self.ctcvr_batchnorm_layer2(dense1)
       dense2 = self.ctcvr_dense_layer2(dense2)
       dense3 = self.ctcvr_batchnorm_layer3(dense2)
       logit = self.ctcvr_logit_output_layer(dense3)
       return tf.keras.activations.sigmoid(logit)
       
    def call(self,inputs):
       user_age_bin = self.user_age_bucket_layer(inputs['user_age'])
       price_bin = self.price_bucket_layer(inputs['price'])
       user_id_embedding = self.user_id_embedding_layer(inputs['user_id'])
       item_id_embedding = self.item_id_embedding_layer(inputs['item_id'])
       price_embedding = self.price_embedding_layer(price_bin)
       age_embedding = self.user_age_embedding_layer(user_age_bin)
       features = [user_id_embedding,item_id_embedding,price_embedding,age_embedding]
       features = tf.keras.layers.Concatenate(axis=-1)(features)
       pctr = self.build_ctr(features)
       pctcvr = self.build_ctcvr(features)
       return pctr,pctcvr
def ctr_loss(label,logits):
   label = tf.where(label == 1,1,0)
   label = tf.cast(label, dtype=tf.float32)
   loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits)
   return  tf.reduce_mean(loss)

def ctcvr_loss(label,logits):
   label = tf.where(label == 2,1,0)
   label = tf.cast(label, dtype=tf.float32)
   loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits)
   return  tf.reduce_mean(loss)

model = ESMM()
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt,
                 loss=[ctr_loss, ctcvr_loss], loss_weights=[1.0, 1.0],
                 metrics=[tf.keras.metrics.AUC()])

model.fit(df_to_dataset(train),validation_data = df_to_dataset(test),epochs= 4)
