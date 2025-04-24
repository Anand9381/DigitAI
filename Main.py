import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw

# Load MNIST dataset and train the CNN model
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

# Create the GUI
class DigitRecognizerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Handwritten Digit Recognizer")
        self.master.geometry("300x350")
        
        self.canvas = tk.Canvas(self.master, width=200, height=200, bg="white")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        self.button_recognize = tk.Button(self.master, text="Recognize", command=self.recognize_digit)
        self.button_recognize.grid(row=1, column=0, pady=10)

        self.clear_button = tk.Button(self.master, text="Clear", command=self.clear_canvas)
        self.clear_button.grid(row=2, column=0, pady=5)

        self.drawing_image = Image.new("L", (200, 200), color=255)
        self.draw = ImageDraw.Draw(self.drawing_image)
        
        self.last_x = None
        self.last_y = None

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=5, fill="black", capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw.line([self.last_x, self.last_y, x, y], fill=0, width=5)
        self.last_x = x
        self.last_y = y

    def on_button_release(self, event):
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.drawing_image = Image.new("L", (200, 200), color=255)
        self.draw = ImageDraw.Draw(self.drawing_image)

    def recognize_digit(self):
        img = self.drawing_image.resize((28, 28))
        img = np.array(img)
        img = img.reshape((1, 28, 28, 1))
        img = img.astype("float32") / 255.0
        
        prediction = model.predict(img)
        predicted_digit = np.argmax(prediction)
        
        messagebox.showinfo("Prediction", f"The predicted digit is: {predicted_digit}")

# Run the GUI
root = tk.Tk()
app = DigitRecognizerApp(root)
root.mainloop()
