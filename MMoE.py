import tensorflow as tf 
import numpy as np  
import pandas as pd 
from sklearn.model_selection import train_test_split
import os

data_dict = {'user_id':np.random.randint(1000,1100,size = 10000),
             'user_age':np.random.randint(1,99,size = 10000),
             'item_id':np.random.randint(1000,1100,size =10000),
             'price':np.random.uniform(1000,1100,size =10000),
             'label':np.random.randint(0,3,size = 10000)}
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
  ds = ds.batch(batch_size, drop_remainder=True)
  return ds
train, test = train_test_split(df, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

class Expert(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.first_batchnorm_layer = tf.keras.layers.BatchNormalization()
        self.first_dense_layer = tf.keras.layers.Dense(64,activation = 'relu')
        self.second_batchnorm_layer = tf.keras.layers.BatchNormalization()
        self.second_dense_layer = tf.keras.layers.Dense(32,activation = 'relu')
    def call(self,features):
        dense1 = self.first_batchnorm_layer(features)
        dense1 = self.first_dense_layer(dense1)
        dense2 = self.second_batchnorm_layer(dense1)
        return self.second_dense_layer(dense2)
class Gate(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.first_batchnorm_layer = tf.keras.layers.BatchNormalization()
        self.first_dense_layer = tf.keras.layers.Dense(64,activation = 'relu')
        self.second_batchnorm_layer = tf.keras.layers.BatchNormalization()
        self.second_dense_layer = tf.keras.layers.Dense(32,activation = 'relu')
        self.third_batchnorm_layer = tf.keras.layers.BatchNormalization()
        ## output_size = expert_number
        self.output_layer = tf.keras.layers.Dense(3,activation = 'softmax')
    def call(self,features):
        dense1 = self.first_batchnorm_layer(features)
        dense1 = self.first_dense_layer(dense1)
        dense2 = self.second_batchnorm_layer(dense1)
        dense2 = self.second_dense_layer(dense2)
        dense3 = self.third_batchnorm_layer(dense2)
        return self.output_layer(dense3)

class Tower(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.first_batchnorm_layer = tf.keras.layers.BatchNormalization()
        self.first_dense_layer = tf.keras.layers.Dense(64,activation = 'relu')
        self.second_batchnorm_layer = tf.keras.layers.BatchNormalization()
        self.second_dense_layer = tf.keras.layers.Dense(32,activation = 'relu')
        self.third_batchnorm_layer = tf.keras.layers.BatchNormalization()
        self.output_layer = tf.keras.layers.Dense(1,activation = 'sigmoid')
    def call(self,features):
        dense1 = self.first_batchnorm_layer(features)
        dense1 = self.first_dense_layer(dense1)
        dense2 = self.second_batchnorm_layer(dense1)
        dense2 = self.second_dense_layer(dense2)
        dense3 = self.third_batchnorm_layer(dense2)
        return self.output_layer(dense3)


class MMoE(tf.keras.Model):
    def __init__(self):
        super().__init__()
        ## bucket_layer
        self.user_age_bucket_layer = tf.keras.layers.Discretization(bin_boundaries=np.arange(1,100,10).tolist())
        self.price_bucket_layer = tf.keras.layers.Discretization(bin_boundaries=np.arange(1000,1100,10).tolist())
        ## embedding_layer 
        self.user_id_embedding_layer = tf.keras.layers.Embedding(input_dim = 101,output_dim = 32)
        self.item_id_embedding_layer = tf.keras.layers.Embedding(input_dim = 101,output_dim = 32)
        self.user_age_embedding_layer = tf.keras.layers.Embedding(input_dim = 11, output_dim=8)
        self.price_embedding_layer = tf.keras.layers.Embedding(input_dim = 11,output_dim = 8)

        ##network
        ## gate_numer = tower_numer =  task_number
        self.expert_list = [Expert() for i in range(3)]
        self.gate_list = [Gate() for i in range(2)]
        self.tower_list = [Tower() for i in range(2)]
    @tf.function
    def call(self,inputs):
        user_age_bin = self.user_age_bucket_layer(inputs['user_age'])
        price_bin = self.price_bucket_layer(inputs['price'])
        user_id_embedding = self.user_id_embedding_layer(inputs['user_id'])
        item_id_embedding = self.item_id_embedding_layer(inputs['item_id'])
        price_embedding = self.price_embedding_layer(price_bin)
        age_embedding = self.user_age_embedding_layer(user_age_bin)
        features = [user_id_embedding,item_id_embedding,price_embedding,age_embedding]
        features = tf.keras.layers.Concatenate(axis=-1)(features)

        batch_size = tf.shape(features)
        ## expert predict
        expert_output_list = [i(features) for i in self.expert_list]
        ## gate predict
        gate_output_list = [i(features) for i in self.gate_list]

        ## tower input  = gate_output_prob (batch_size,1) * expert_output_dense (batch_size,32)
        tower_input_list = []
        for i in gate_output_list:
            split_vectors = tf.split(i, num_or_size_splits=3, axis=1)
            ## initial temp for add
            ## batch_size(32) x expert output size(32)
            tmp = tf.zeros((32, 32))
            for j in range(len(split_vectors)):
                 tmp = tf.add(tmp,split_vectors[j] * expert_output_list[j])
            tower_input_list.append(tmp)    
        ##tower predict
        predict_list = []
        for i in range(len(self.tower_list)):
            predict_list.append(self.tower_list[i](tower_input_list[i]))
        return predict_list[0],predict_list[1]

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

model = MMoE()
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt,
                 loss=[ctr_loss, ctcvr_loss], loss_weights=[1.0, 1.0],
                 metrics=[tf.keras.metrics.AUC()])

model.fit(df_to_dataset(train),validation_data = df_to_dataset(test),epochs= 4)
        
    
