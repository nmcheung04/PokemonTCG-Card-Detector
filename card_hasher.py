import imagehash
from PIL import Image
import pickle
import os

def get_image(image_folder):
    '''
    Gets PNG images from a folder containing images
    '''
    for filename in os.listdir(image_folder):
        if filename.endswith('.png'):
            file_path = os.path.join(image_folder, filename)
            with Image.open(file_path) as img:
                yield filename, img.copy()

def perceptual_hasher(card_database, filename, img):
    '''
    Computes perceptual hash of a given image and adds it to the card database
    '''
    img_hash = imagehash.phash(img)
    name_without_extension = os.path.splitext(filename)[0]
    card_database[name_without_extension] = str(img_hash)
    print(f"Added {name_without_extension} to database with hash {img_hash}")

def main(image_folder = "card_images", output_file = "card_database.pkl"):
    '''
    Process all images in a folder and save the database to a pickle file
    '''
    if not os.path.isdir(image_folder):
        print(f"Folder {image_folder} does not exist")
        return

    card_database = {}

    for filename, img in get_image(image_folder):
        perceptual_hasher(card_database, filename, img)

    with open(output_file, 'wb') as db_file:
        pickle.dump(card_database, db_file)
        print(f"Sucessefully saved database to {output_file}")

    print(len(card_database))

if __name__ == "__main__":
    main()