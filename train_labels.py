import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ✅ Step 1: Create data generator
datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

# ✅ Step 2: Load training data
train_generator = datagen.flow_from_directory(
    "your_dataset_folder",         # <--- 🔁 Change this to your dataset path
    target_size=(224, 224),        # Use the same size used for training your model
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# ✅ Step 3: Save class label mapping
with open("class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f)

print(f"[✅] Saved label index mapping: {train_generator.class_indices}")