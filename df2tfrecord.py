import tensorflow as tf 
import pandas as pd 
import numpy as np 
def create_tf_example(row):
    features = {
        'label':tf.train.Feature(int64_list=tf.train.Int64List(value = [int(row['label'])])),
        'user_age':tf.train.Feature(float_list = tf.train.FloatList(value = [row['user_age']])),
        'item_id':tf.train.Feature(float_list = tf.train.FloatList(value = [row['item_id']])),
        'price':tf.train.Feature(float_list = tf.train.FloatList(value = [row['price']])),
        'user_id':tf.train.Feature(float_list = tf.train.FloatList(value = [row['user_id']])),                            
    }
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example
def df2tfrecord(df):
    df['label'] = df['label'].astype('int64')
    df['user_age'] = df['user_age'].astype('float32')
    df['price'] = df['price'].astype('float32')
    df['user_id'] = df['user_id'].astype('float32')
    df['item_id'] = df['item_id'].astype('float32')
    tf_examples = df.apply(lambda row: create_tf_example(row), axis=1)
    with tf.io.TFRecordWriter('example.tfrecord') as writer:
        for example in tf_examples:
            writer.write(example.SerializeToString())
if __name__ == '__main__':
    data_dict = {'user_id':np.random.randint(1000,1100,size = 100),
             'user_age':np.random.randint(1,99,size = 100),
             'item_id':np.random.randint(1000,1100,size =100),
             'price':np.random.uniform(1000,1100,size =100),
             'label':np.random.randint(0,2,size = 100)}
    df = pd.DataFrame(data_dict)

    print(df.dtypes)
    df2tfrecord(df)

