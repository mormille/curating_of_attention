import os
from numba import vectorize, jit, cuda
from shutil import copyfile

os.chdir('/home/luizmormille/Documents/AtsumiLab/Self-supervised-Learning/Vanilla/')
print(os.path.abspath(os.sep))
#!git clone https://github.com/jaddoescad/ants_and_bees.git

#@cuda.jit
def separate_images():
    file_count = 0
    for folder in os.listdir('ants_and_bees/'):
        #break
        if not folder.startswith('.'):
            print(folder)
            for file_folder in os.listdir('ants_and_bees/' + folder):
                if not file_folder.startswith('.'):
                    print(file_folder)
                    for filename in os.listdir('ants_and_bees/' + folder + '/' + file_folder):
                        file_count += 1
                        folder_name = 'self_supervised_exemplar/val/' + str(file_count)
                        if not os.path.exists(folder_name):
                            os.makedirs(folder_name)
                            copyfile('ants_and_bees/' + folder + '/' + file_folder + '/' + filename, folder_name + '/' + filename)
                        #    break  
    print(file_count)

import random
from scipy import ndarray
import skimage as sk
from skimage import io
from skimage import transform
from skimage import util
import time

def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]


def augmentation():
    since = time.time()
    for folder in os.listdir('self_supervised_exemplar/val/'):
        #print(folder)
        #break
        folder_path = 'self_supervised_exemplar/val/' + folder + '/'
        # the number of file to generate
        num_files_desired = 20

    # dictionary of the transformations we defined earlier
        available_transformations = {
            'rotate': random_rotation,
            'noise': random_noise,
            'horizontal_flip': horizontal_flip
        }

        # find all files paths from the folder
        images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        num_generated_files = 0
        while num_generated_files <= num_files_desired:
            # random image from the folder
            image_path = random.choice(images)
            # read image as an two dimensional array of pixels
            image_to_transform = sk.io.imread(image_path)
            # random num of transformation to apply
            num_transformations_to_apply = random.randint(1, len(available_transformations))

            num_transformations = 0
            transformed_image = None
            while num_transformations <= num_transformations_to_apply:
                # random transformation to apply for a single image
                key = random.choice(list(available_transformations))
                transformed_image = available_transformations[key](image_to_transform)
                num_transformations += 1

                new_file_path = '%s/augmented_image_%s.jpg' % (folder_path, num_generated_files)

                # write image to the disk
                io.imsave(new_file_path, transformed_image)
                num_generated_files += 1
                #break
        print(folder)

    time_elapsed = time.time() - since
    print('Data Preprocessing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

separate_images()
augmentation()