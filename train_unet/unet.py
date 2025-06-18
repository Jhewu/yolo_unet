#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Libraries
import os
os.environ["KERAS_BACKEND"] = "tensorflow" 

import keras
import numpy as np
import tensorflow as tf
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io
from keras import layers
import cv2 as cv


# In[2]:


# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Limiting GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)


# In[3]:


# Hyperparameters

image_dir = "cleaned_t1c/images"
mask_dir = "cleaned_t1c/masks"
image_size = (240, 240)
batch_size = 16


# In[4]:


# Create lists of images for each partition

def sort_list(image_dir): 
    # Create function that will sort the partition list
    return sorted(
        [
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.endswith(".png")
        ]
        )

# Create the sorted list for the images partition
train_image = sort_list(os.path.join(image_dir, "train"))
val_image = sort_list(os.path.join(image_dir, "val"))
test_image = sort_list(os.path.join(image_dir, "test"))

# Create the sorted list for the masks partition
train_mask = sort_list(os.path.join(mask_dir, "train"))
val_mask = sort_list(os.path.join(mask_dir, "val"))
test_mask = sort_list(os.path.join(mask_dir, "test"))

print(f"\nThere is approximately {len(train_image)} images within the training set\n")

for input_path, target_path in zip(train_image[:10], train_mask[:10]):
    print(input_path, "|", target_path)


# In[5]:


# Display the image to ensure it works correctly

from IPython.display import Image, display
from keras.utils import load_img
from PIL import ImageOps

# Display input image #7
display(Image(filename=train_image[100]))

# Display auto-contrast version of corresponding target (per-pixel categories)
img = ImageOps.autocontrast(load_img(train_mask[100]))

display(img)


# In[6]:


# Create the function that returns TF datasets by pairing the image and mask

def get_dataset(
    batch_size,
    image_size,
    image_paths,
    mask_paths,):
    """Returns a TF Dataset."""

    def load_img_masks(input_img_path, target_img_path):
        input_img = tf_io.read_file(input_img_path)
        input_img = tf_io.decode_png(input_img, channels=1)
        input_img = tf_image.resize(input_img, image_size)
        input_img = tf.cast(input_img, tf.float32) / 255.0

        target_img = tf_io.read_file(target_img_path)
        target_img = tf_io.decode_png(target_img, channels=1)
        target_img = tf_image.resize(target_img, image_size, method="nearest")
        target_img = tf.cast(target_img, tf.float32) / 255.0

        return input_img, target_img

    dataset = tf_data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(load_img_masks, num_parallel_calls=tf_data.AUTOTUNE)
    return dataset.batch(batch_size)


# In[7]:


# Define the function to create a unet

def get_model(img_size):
    inputs = keras.Input(shape=img_size + (1,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(1, 1, activation="sigmoid", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


# Build model
model = get_model(image_size)
model.summary()


# In[8]:


# Create the train, val and test split with the pairings

train_dataset = get_dataset(
    batch_size,
    image_size,
    train_image,
    train_mask,)

val_dataset = get_dataset(
    batch_size, 
    image_size, 
    val_image, 
    val_mask, )

test_dataset = get_dataset(
    batch_size, 
    image_size, 
    test_image, 
    test_mask, )

# Check values in the first batch
for input_img, target_img in train_dataset.take(10):
    print("Information of image:\n", np.max(input_img), np.min(input_img), input_img.dtype)
    print("Information of mask:\n", np.max(target_img), np.min(target_img), target_img.dtype, np.unique(target_img))


# In[9]:


def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def weighted_binary_crossentropy(beta=0.9):
    def loss(y_true, y_pred):
        weight_for_1 = beta
        weight_for_0 = 1 - beta
        
        y_true = tf.cast(y_true, tf.float32)
        
        loss = -(weight_for_1 * y_true * tf.math.log(y_pred + 1e-7) +
                 weight_for_0 * (1 - y_true) * tf.math.log(1 - y_pred + 1e-7))
        
        return tf.reduce_mean(loss)
    return loss


# In[10]:


# loss_fn = dice_loss if loss_type == 'dice' else weighted_binary_crossentropy(beta=0.9)

class_weights = {0: 1.0, 1: 10.0}

# Configure the model for training.
# We use the "sparse" version of categorical_crossentropy
# because our target data is integers.
model.compile(
    optimizer=keras.optimizers.Adam(1e-4), 
    loss=dice_loss, 
)

callbacks = [
    keras.callbacks.ModelCheckpoint("oxford_segmentation.keras", save_best_only=True),
    keras.callbacks.CSVLogger(filename="cvslog.csv", separator=",", append=True)
]

# Train the model, doing validation at the end of each epoch.
epochs = 50
model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=val_dataset,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=2,
)


# In[11]:


# Generate predictions for all images in the validation set
import cv2 as cv

test_prediction = model.predict(test_dataset)

def display_mask(i):
    """Quick utility to display a model's prediction."""
    mask = np.argmax(test_prediction[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = ImageOps.autocontrast(keras.utils.array_to_img(mask))
    display(img)
    cv.imwrite("mask.png", mask)

# Display results for validation image #10
i = 100

# Display input image
display(Image(filename=test_image[i])) # test_image is a list
img = cv.imread(test_image[i])
cv.imwrite("original.png", img)

# Display ground-truth target mask
img = ImageOps.autocontrast(load_img(test_mask[i])) # test_mask is also a list
display(img)
img = cv.imread(test_mask[i])
cv.imwrite("ground_truth.png", img)

# Display mask predicted by our model
display_mask(i)  # Note that the model only sees inputs at 150x150.


# In[12]:


# obtain the predictions
train_prediction = model.predict(train_dataset)


# In[13]:


# Generate predictions for all images in the validation set
import cv2 as cv

def display_mask(i):
    """Quick utility to display a model's prediction."""
    mask = np.argmax(train_prediction[i], axis=-1) 
    # print(np.unique(mask))
    # print(mask.shape)
    mask = np.expand_dims(mask, axis=-1)
    mask = mask * (255 // (np.max(mask) + 1))  # Normalize the mask to [0, 255]
    img = ImageOps.autocontrast(keras.utils.array_to_img(mask))
    display(img)
    cv.imwrite("mask.png", mask)

# Display results for training image
i = 99

# # Display input image
# display(Image(filename=train_image[i])) 
# img = cv.imread(train_image[i])
# cv.imwrite("original.png", img)

# Display ground-truth target mask
img = ImageOps.autocontrast(load_img(train_mask[i])) # test_mask is also a list
display(img)
img = cv.imread(train_mask[i])
cv.imwrite("ground_truth.png", img)

# Display mask predicted by our model
display_mask(i)  # Note that the model only sees inputs at 150x150.


# In[ ]:


# Display results for validation image #10
i = 200

# # Display input image
display(Image(filename=test_image[i])) # test_image is a list

# # Display ground-truth target mask
img = ImageOps.autocontrast(load_img(test_mask[i])) # test_mask is also a list
display(img)

# # Display mask predicted by our model
# display_mask(i)  # Note that the model only sees inputs at 150x150.

og_img = cv.imread(test_image[i])
mask_img = cv.imread(test_mask[i])

multiplied = og_img*mask_img

cv.imshow("multiplied", multiplied)
# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv.waitKey(0)

# closing all open windows
cv.destroyAllWindows()


# In[ ]:


# Add this after loading a batch
for images, masks in train_dataset.take(5):
    print("Images shape:", images.shape)
    print("Images min/max:", tf.reduce_min(images), tf.reduce_max(images))
    print("Masks shape:", masks.shape)
    print("Unique mask values:", np.unique(masks))

