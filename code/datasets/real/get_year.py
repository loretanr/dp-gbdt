
from urllib.request import urlretrieve
from zipfile import ZipFile
import os
import progressbar

pbar = None

def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None


filename = 'YearPredictionMSD.txt.zip'

get_year_url = ('https://archive.ics.uci.edu/ml/machine-learning-databases/00203/' + filename)
filename_real_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)

nothing_todo = True

if not (os.path.isfile(filename_real_path) or os.path.isfile(filename_real_path.replace('.zip', ''))):
    print("downloading YearPredictionMSD.txt.zip dataset:")
    urlretrieve(get_year_url, filename_real_path, show_progress)
    nothing_todo = False

if not os.path.isfile(filename_real_path.replace('.zip', '')):
    print("Extracting " + filename)
    zip = ZipFile(filename_real_path)
    zip.extractall()
    nothing_todo = False

if(nothing_todo):
    print("dataset YearPredictionMSD found")