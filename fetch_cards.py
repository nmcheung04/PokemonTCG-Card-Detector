import requests
import time
import os
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables once at the start
load_dotenv()

def download_image(image_url, save_path):
    '''
    Downloads an image from a given url and saves it to the specified path
    '''
    response = requests.get(image_url, stream = True)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        tqdm.write(f"Image successfully downloaded: {save_path}")
    else:
        tqdm.write(f"Failed to download image from {image_url}")

def fetch_and_process(api_url, headers, params, save_dir):
    '''
    Fetches image using API and saves it to save_dir
    '''
    current_page = 1
    page_size = params.get('pageSize', 250)

    while True:
        params['page'] = current_page
        response = requests.get(api_url, headers=headers, params=params)

        if response.status_code == 429:
            print("Rate limited. Sleeping for 10 seconds...")
            time.sleep(10)
            continue
        elif response.status_code != 200:
            print(f"Failed to fetch data: {response.status_code}")
            break

        data = response.json()
        cards = data.get('data', [])

        if not cards:
            break

        for card in tqdm(cards, desc=f"Page {current_page}", unit="card"):
            name = card['name'].replace(' ', '_').replace('/', '_')
            card_id = card['id']
            image_url = card['images']['large']
            file_name = f"{name}_{card_id}.png"
            save_path = os.path.join(save_dir, file_name)
            download_image(image_url, save_path)

        if len(cards) < page_size:
            break

        current_page += 1

def main(save_dir = "card_images"):
    api_url = "https://api.pokemontcg.io/v2/cards"
    api_key = os.getenv("POKEMON_API_KEY")
    headers = {
        "X-Api-Key": api_key
    }
    params = {
        "pageSize": 250,
        "page" : 1
    }
    os.makedirs(save_dir, exist_ok = True)
    fetch_and_process(api_url, headers, params, save_dir)

if __name__ == "__main__":
    main()