from pathlib import Path
from fastai.vision.all import *
from duckduckgo_search import ddg_images
from fastdownload import download_url
from time import sleep

class ImageCollectionController:
    def run_image_classification(self):
        def search_images(term, max_images=100):
            return L(ddg_images(term, max_results=max_images)).itemgot('image')

        def download_and_save_images(search_term, destination, max_images=100, retries=1):
            print('download and process images')
            urls = search_images(search_term, max_images)
            for i, url in enumerate(urls):
                filename = f"{search_term}_{i}.jpg"
                attempt = 0
                while attempt < retries:
                    try:
                        download_url(url, destination / filename, show_progress=False)
                        break
                    except Exception as e:
                        print(f"Error downloading {url}: {e}, attempt {attempt + 1} of {retries}")
                        attempt += 1
                        sleep(5)

        searches = ['bird', 'forest']
        path = Path('data_set')
        path.mkdir(exist_ok=True, parents=True)

        for term in searches:
            dest = path / term
            dest.mkdir(exist_ok=True, parents=True)
            if not list(dest.glob('*.jpg')):  # Check if the folder is empty
                print(f'Downloading images for: {term}')
                download_and_save_images(term, dest, 200)
                sleep(10)  # To avoid overloading the server
                resize_images(dest, max_size=400, dest=dest)
            else:
                print(f'Images already downloaded for: {term}')

        print('start removing failed images')
        failed = verify_images(get_image_files(path))
        failed.map(Path.unlink)
        return 'OK'
