import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import sys

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from sklearn.metrics import confusion_matrix, classification_report

epochs = int(sys.argv[1])
activation_function = str(sys.argv[2])
n_layers = int(sys.argv[3])

dir_name = "./output/{}_{}_{}".format(epochs, activation_function, n_layers)
base_filename = "{}/{}_{}_{}".format(dir_name, epochs, activation_function, n_layers)

if not os.path.exists("./output"):
  os.mkdir("./output")

if not os.path.exists(dir_name):
  os.mkdir(dir_name)

batch_size = 64
img_height = 32
img_width = 32
data_dir = "./myData"
labels_path="./labels.csv"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

num_classes = len(train_ds.class_names)

classID = train_ds.class_names
classIDLabels = pd.read_csv(labels_path)
classLabels=[]
for i in classID:
    classLabels.append(classIDLabels['Name'][int(i)])

model = Sequential()
model.add(layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)))
filters=16
for i in range(n_layers):
  model.add(layers.Conv2D(filters, 3, padding='same', activation=activation_function))
  model.add(layers.MaxPooling2D())
  filters=filters*2
model.add(layers.Flatten())
model.add(layers.Dense(filters, activation=activation_function))
model.add(layers.Dense(num_classes))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig("{}{}".format(base_filename,"_train.png"))

f=open("{}{}".format(base_filename,".txt"),"w")
f.write("Some sample predictions\n\n")
print("Some sample predictions saved to file")
max_display = 10
current_display = 0
testLabel=[]
predLabel=[]
for images, labels in val_ds:
    predictions = model.predict(images)
    for i in range(len(predictions)):
        scores = tf.nn.softmax(predictions[i])
        idx = np.argmax(scores)
        acc = np.max(scores) * 100
        testLabel.append(classLabels[labels[i]])
        predLabel.append(classLabels[idx])
        if current_display < max_display:
            f.write("Predicted: {}\nActual: {}\nAccuracy: {:.2f}%\n\n".format(classLabels[idx],classLabels[labels[i]],acc))
        current_display+=1

confusion = confusion_matrix(testLabel, predLabel)

f.write("\n\nClassification report\n\n")
f.write(classification_report(testLabel, predLabel, target_names=classLabels))
f.close()

plt.figure(figsize=(10,10))
sns.heatmap(confusion, annot = True, cbar = False, cmap='Paired', fmt="d", xticklabels=classLabels, yticklabels=classLabels)
plt.savefig("{}{}".format(base_filename,"_confusion_matrix.png"), bbox_inches='tight')
