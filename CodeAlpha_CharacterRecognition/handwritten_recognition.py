import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ==============================
# 1️⃣ Load MNIST Dataset
# ==============================
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values (0-255 → 0-1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape for CNN (add channel dimension)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# ==============================
# 2️⃣ Build CNN Model
# ==============================
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==============================
# 3️⃣ Train Model
# ==============================
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# ==============================
# 4️⃣ Evaluate Model
# ==============================
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# ==============================
# 5️⃣ Save Model
# ==============================
model.save("handwritten_model.h5")

# ==============================
# 6️⃣ Predict One Image
# ==============================
index = 5
plt.imshow(X_test[index].reshape(28,28), cmap='gray')
plt.title("True Label: " + str(y_test[index]))
plt.show()

prediction = model.predict(X_test[index].reshape(1,28,28,1))
print("Predicted Digit:", np.argmax(prediction))
