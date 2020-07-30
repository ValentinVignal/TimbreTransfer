import tensorflow as tf
import tensorflow_datasets as tfds

ds = tfds.load('nsynth', split='train', shuffle_files=True)

ds = ds.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
for example in ds.take(1):
    print(len(example['audio']), '-', example['instrument'])