import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import os

# Step 1: Load and Inspect the Data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')
sample_submissions = pd.read_csv('sample_submissions.csv')

print(train_data.head())
print(test_data.head())
print(sample_submissions.head())

# Step 2: Preprocess the Data
def preprocess_train_images(df, img_dir):
    images = []
    for index, row in df.iterrows():
        label = row['label']
        file_name = row['file_name']
        img_path = os.path.join(img_dir, str(label), file_name)
        print(img_path)  # Debugging line
        if os.path.exists(img_path):
            img = load_img(img_path, target_size=(224, 224))  # Resizing the image
            img_array = img_to_array(img) / 255.0  # Normalize pixel values
            images.append(img_array)
        else:
            print(f"File not found: {img_path}")
    return np.array(images)

def preprocess_test_images(df, img_dir):
    images = []
    for index, row in df.iterrows():
        file_name = row['file_name']
        img_path = os.path.join(img_dir, file_name)
        print(img_path)  # Debugging line
        if os.path.exists(img_path):
            img = load_img(img_path, target_size=(224, 224))  # Resizing the image
            img_array = img_to_array(img) / 255.0  # Normalize pixel values
            images.append(img_array)
        else:
            print(f"File not found: {img_path}")
    return np.array(images)

# Preprocess images in the train and test folders
train_images = preprocess_train_images(train_data, 'data/train')
test_images = preprocess_test_images(test_data, 'data/test')

# Step 3: Encode Labels
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_data['label'])

# Step 4: Build the Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 5: Train the Model
model.fit(train_images, train_labels, epochs=10)  # Removed validation_split

# Step 6: Make Predictions
predictions = model.predict(test_images)
predicted_labels = (predictions > 0.5).astype(int)

# Step 7: Prepare Submission
submission = pd.DataFrame({
    'file_name': test_data['file_name'],
    'label': predicted_labels.flatten()
})
submission.to_csv('final_submission.csv', index=False)


