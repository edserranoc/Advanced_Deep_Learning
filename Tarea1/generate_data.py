# load libraries
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Set a fixed seed for reproducibility
np.random.seed(42)

# Directory and file paths
data_dir = './data'
generate_data_dir = './generate_data'
data_file = os.path.join(data_dir, 'codemaps.npy')
labels_file = os.path.join(data_dir, 'labels.npy')

# Directory to save the generated splits
if not os.path.exists(generate_data_dir):
    os.makedirs(generate_data_dir)

train_data_file = os.path.join(generate_data_dir, 'X_train.npy')
val_data_file = os.path.join(generate_data_dir, 'X_val.npy')
test_data_file = os.path.join(generate_data_dir, 'X_test.npy')
train_labels_file = os.path.join(generate_data_dir, 'y_train.npy')
val_labels_file = os.path.join(generate_data_dir, 'y_val.npy')
test_labels_file = os.path.join(generate_data_dir, 'y_test.npy')

# Check if the splits already exist, if not generate and save them
if not (os.path.exists(train_data_file) and os.path.exists(val_data_file) and os.path.exists(test_data_file)):
    # Load codemaps and labels from the data folder
    data = np.load(data_file)
    labels = np.load(labels_file)

    # Data preprocessing
    data = data.astype('float32') / 255.0
    data = data.reshape(-1, 12 * 12)
    
    # Split the data into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    # Save the split datasets
    np.save(train_data_file, X_train)
    np.save(val_data_file, X_val)
    np.save(test_data_file, X_test)
    np.save(train_labels_file, y_train)
    np.save(val_labels_file, y_val)
    np.save(test_labels_file, y_test)

    print("Data splits generated and saved.")
else:
    print("Data splits were  generated previously.")
