from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
import glob
import pandas as pd
import shutil


def get_generators(dataset, batch_size, pretrained, train_augmentation):
    """
    Get data generators for the specified dataset.

    Parameters:
    - dataset (str): Name of the dataset ("cifar10", "tiny-imagenet", or "pets").
    - batch_size (int): Batch size for the generators.
    - pretrained (bool): Flag indicating whether a pretrained model is being used.
    - train_augmentation (bool): Flag indicating whether to apply data augmentation during training.

    Returns:
    - train_generator (tf.keras.utils.Sequence): Data generator for training set.
    - validation_generator (tf.keras.utils.Sequence): Data generator for validation set.
    - test_generator (tf.keras.utils.Sequence): Data generator for test set.
    """

    # Define ImageDataGenerators for training and validation
    if train_augmentation == True:
        # Data augmentation for training set if True
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )
    else:
        # No data augmentation for training set
        train_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2
            )

    test_datagen = ImageDataGenerator(rescale=1./255)
    
    if dataset == "cifar10" :
        # A specific location where CIFAR would be downloaded (automatically) for the 1st time
        base_dir = '../Datasets/CIFAR/' 
        
        x_train_path, x_test_path = base_dir+'cifar10_train/', base_dir+'cifar10_test/'

        if not os.path.exists(x_train_path):
            # Downloads and put cifar into correct directories.
            generate_data_directories(dataset, base_dir)
        
        # If we have an Imagenet pretrained model, we need to upscale cifar
        if pretrained:
            h,w = 224,224
            print("\nUpscaling Cifar from (32,32) to (224,224) since ImageNet Pretrained Model is being used")
        else:
            # otherwise we just use 32,32 resolution
            h,w = 32,32
            print("Using original resolution of (32,32) for Cifar-10")
        
        # We flow the images through a data generator while specifying the target size and batch size
        print("Train set :")
        train_generator = train_datagen.flow_from_directory(x_train_path, target_size=(h,w), batch_size=batch_size,
                                                            subset='training')
        
        # Val subset of train_datagen
        print("Val set :")
        validation_generator = train_datagen.flow_from_directory(x_train_path, target_size=(h,w),
                                                                 batch_size=batch_size,
                                                                 subset='validation')
        # Test datagen creates the test set.
        print("Test set :")
        test_generator = test_datagen.flow_from_directory(x_test_path, target_size=(h,w), batch_size=batch_size)
        
        print("\nLoaded Cifar 10 Dataset\n")
    
    elif dataset == "tiny-imagenet":
        # A specific location where dataset should be downloaded
        base_dir = '../Datasets/tiny-imagenet-200'
          
        train_dir = os.path.join(base_dir, 'train')
        test_dir = os.path.join(base_dir, 'val')
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"The training directory {train_dir} does not exist. Go to https://www.image-net.org/download.php and make an account. Download tiny-imagenet-200.zip and unzip it at ../Datasets/")

        # Check if testing directory exists
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"The testing directory {test_dir} does not exist.")
        
        # Reorganize the directory so we can load it later
        generate_data_directories(dataset, base_dir)
        
        img_width, img_height = 224,224

        # Flow images in batches using train_datagen generator
        print("Train set :")
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical',
        subset='training')

        print("Val set :")
        validation_generator = train_datagen.flow_from_directory(train_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation') 


        # Flow images in batches using test_datagen generator
        print("Test set :")
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')
        
        print("\nLoaded Tiny Imagenet Dataset\n")

    elif dataset=="pets":
        # A specific location where dataset should be downloaded
        base_dir = '../Datasets/PETS/images/'

        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"The directory {base_dir} does not exist. \
                                    \nPlease extract the 'images' directory from Pets dataset, found at Link : 'https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz' into ../Datasets/PETS/")

        # We need a dataframe to load the images since they are all in 1 directory
        train_df, test_df = get_dataframes(dataset, base_dir)

        print("Train set :")
        train_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
                                                    directory=base_dir,
                                                    x_col='file_name',
                                                    y_col='label',
                                                    target_size=(224, 224),
                                                    class_mode='categorical',
                                                    batch_size=batch_size,
                                                    subset='training')
        
        print("Val set :")
        validation_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
                                                                directory=base_dir,
                                                                x_col='file_name',
                                                                y_col='label',
                                                                target_size=(224, 224),
                                                                class_mode='categorical',
                                                                batch_size=batch_size,
                                                                subset='validation')
        print("Test set :")
        test_generator = test_datagen.flow_from_dataframe(dataframe=test_df,
                                                        directory=base_dir,
                                                        x_col='file_name',
                                                        y_col='label',
                                                        target_size=(224, 224),
                                                        class_mode='categorical',
                                                        batch_size=batch_size)
        
        print("\nLoaded Pets Dataset\n")

    return train_generator, validation_generator, test_generator


def generate_data_directories(dataset, base_dir):
    if dataset=="cifar10":
        
        print("\nDownloading Cifar Data and putting into directories [First Time only]\n")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        
        # Specify the directories to save the datasets
        save_dir_train_x = base_dir+'cifar10_train/'
        save_dir_test_x = base_dir+'cifar10_test/'

        # Create directories if they don't exist
        for i in range(10):
            os.makedirs(f"{save_dir_train_x}/{i}", exist_ok=True)
            os.makedirs(f"{save_dir_test_x}/{i}", exist_ok=True)

        # Save training dataset
        for i in range(len(x_train)):
            image_filename = os.path.join(save_dir_train_x, f"{y_train[i][0]}/image_{i}.png")
            tf.keras.preprocessing.image.save_img(image_filename, x_train[i])

        # Save testing dataset
        for i in range(len(x_test)):
            image_filename = os.path.join(save_dir_test_x, f"{y_test[i][0]}/image_{i}.png")
            tf.keras.preprocessing.image.save_img(image_filename, x_test[i])
        
        

    elif dataset=="tiny-imagenet":
        
        validation_dir = os.path.join(base_dir, 'val')
              
        images_folder_path = os.path.join(validation_dir, 'images')
        
        # tiny imagenet downloads into val/images all the images. We need to make folders for each class.
        if os.path.exists(images_folder_path) and os.path.isdir(images_folder_path):
            
            print("\nRe-organizing tiny-imagenet Val directory for loading [After First Time Download only]")
            val_annotations_path = os.path.join(validation_dir, 'val_annotations.txt')

            # Create a directory structure for validation images based on annotations
            val_annotations = pd.read_csv(val_annotations_path, sep='\t', header=None, names=['File', 'Class', 'X', 'Y', 'H', 'W'])
            val_images_dir = os.path.join(validation_dir, 'images')

            for class_name in val_annotations['Class'].unique():
                new_dir = os.path.join(validation_dir, class_name)
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                    
            # Move the files into appropriate class directories
            for index, row in val_annotations.iterrows():
                file_path = os.path.join(val_images_dir, row['File'])
                new_path = os.path.join(validation_dir, row['Class'], row['File'])
                shutil.move(file_path, new_path)

            # Delete the 'images' folder and 'val_annotations.txt' file
            delete_if_exists(images_folder_path)
            delete_if_exists(val_annotations_path)

            # Delete any hidden folders (sometimes there are hidden folders lurking that increase class count)
            for item in os.listdir(validation_dir):
                item_path = os.path.join(validation_dir, item)
                if os.path.isdir(item_path) and item.startswith('.'):
                    delete_if_exists(item_path)

    return


def delete_if_exists(path):
    """
    Function to delete a directory or file if it exists
    """
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.isfile(path):
        os.remove(path)

def get_dataframes(dataset, dir_name):
    """
    Get train and test dataframes for the specified dataset.

    Parameters:
    - dataset (str): Name of the dataset ("pets").
    - dir_name (str): Directory name where the dataset is located.

    Returns:
    - train_df (pd.DataFrame): DataFrame containing information for the training set.
    - test_df (pd.DataFrame): DataFrame containing information for the test set.
    """
    if dataset=="pets":
        # Get a list of file paths in the specified directory
        pets_files = glob.glob(dir_name+"*")
        
        # Create a DataFrame with the full file paths
        pets_df = pd.DataFrame(pets_files)
        pets_df.columns =['full_path']
        
        # Extract file names and labels from the full file paths
        pets_df['file_name'] = pets_df['full_path'].str.split('images/').str[1]
        pets_df['label'] = pets_df['file_name'].str.rsplit('_',n=1).str[0]

        # Split the DataFrame into training and testing sets
        train_df, test_df = train_test_split(pets_df, test_size=0.1, random_state=42)

        return train_df, test_df
    
