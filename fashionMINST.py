import tensorflow as tf
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt 

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0
# To bring all the values of image pixels between o to 1 i.e Normalisation

plt.imshow(train_images[7], cmap=plt.cm.binary)
plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    # relu is rectified linear function used on mny diff models for fast processing
    keras.layers.Dense(10, activation="softmax")
    # softmax is a functino tha choses output such that the total of all neurons is 1.
    # After that it selects the max value neuron as ans
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# https://www.tensorflow.org/tutorials/keras/classification
# Open the above link for more description

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Tested acc: ", test_acc)

prediction = model.predict(test_images)

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap = plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Predcition: " + class_names[np.argmax(prediction[i])])
    plt.show()
    
