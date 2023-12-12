import re
import urllib3
from bs4 import BeautifulSoup
import time
import random
import os
from datetime import date

charts_url = 'https://www.billboard.com/charts/hot-100/'

# Retrieves the top N songs with their artist from billboard.com
# Returns a list of tuples holding the song title and artist [(title, artist)]
def get_top_songs(amount = 10):
    http = urllib3.PoolManager()
    page = http.request('GET', charts_url)
    soup = BeautifulSoup(page.data, "html.parser")
    top_html = soup.findAll('li', class_= 'lrv-u-width-100p')

    count = 0
    top = []
    for li in top_html:
        song_name_html = li.find('h3', id='title-of-a-story')
        artist_html = li.find('span')
        if song_name_html and count < amount:
            song_name = re.sub(' +', ' ', song_name_html.text).lower().strip()
            artist = re.sub('[^A-Za-z0-9.&]+', ' ', artist_html.text).lower()
            artist = artist.split('&')[0]
            artist = artist.split('featuring')[0]
            top.append((song_name, artist))
            count += 1
    return top

# Constructs the url to the song's lyrics using the song name and artist
def get_lyrics_url(song_name, artist):
    song_name = re.sub('[^A-Za-z0-9.]+','', song_name).lower()
    artist = re.sub('[^A-Za-z0-9.]+','', artist).lower()
    return 'https://www.azlyrics.com/lyrics/' + artist + '/' + song_name + '.html'

# Gets all the songs urls and puts them in a new list
# Returns a list of sets containing the title of the song, the artist, and the url to the lyrics (title, artist, url)
def get_all_lyrics_urls(songs):
    urls = []
    for song in songs:
        urls.append((song[0], song[1], get_lyrics_url(song[0], song[1])))
    return urls

# Scrapes the lyrics of the song from url
# Returns String of lyrics
def get_lyrics(url, song_name, artist):
    http = urllib3.PoolManager()
    page = http.request('GET', url)
    soup = BeautifulSoup(page.data, "html.parser")
    raw = soup.text
    start_pattern = r'' + artist.strip() + r'\s*(&?)? .*\n+"' + song_name.strip() + r'"'
    start_pattern = re.compile(start_pattern, re.IGNORECASE)
    matches = list(start_pattern.finditer(raw))
    if not matches:
        print('--no regex match found' + song_name.strip(), end='')
        return '~~~~~ERROR_DURING_RETRIEVAL~~~~~'
    start = matches[-1].end()
    stop = re.search(r'Submit Corrections', raw).start()
    time.sleep(random.randint(5, 15))
    return raw[start:stop]

# Gets the lyrics of all songs in provided list
# Returns a list of tuples containing the song name and the lyrics in a String [(title, lyrics)]
def get_all_lyrics(songs_and_urls):
    all_lyrics = []
    count = 0
    for song in songs_and_urls:
        count += 1
        print('--' + str(round((count / len(songs_and_urls)) * 100, 2)) + '%: ' + song[0].strip())
        all_lyrics.append([song[0], get_lyrics(song[2], song[0], song[1]).strip()])
    return all_lyrics

# Saves the String of lyrics to a text file from each song in provided list
def save_lyrics(all_lyrics):
    print('saving lyrics to resources...')
    path = './resources/' + str(date.today())
    if not os.path.exists(path):
        os.makedirs(path)
    for lyrics in all_lyrics:
        f = open(path + '/' + lyrics[0].strip() + '.txt', 'w+')
        f.write(lyrics[1].strip())
        print('--' + str(f.tell()))
        f.close()
    print('saved files to ' + path + '/')

def load_corpus_from_saved_files(path='./resources/test/'):
    print('creating corpus from resources...')
    corpus = []
    file_names = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for file_name in file_names:
        data = ''
        file = open(path + file_name, 'r')
        for line in file:
            data += line + '\n'
        corpus.append((file_name, data))
        file.close()
    print('done creating corpus.')
    return corpus

def web_scrape_lyrics(save = False, amount = 20):
    print('scraping for corpus...')
    top_songs = get_top_songs(amount)
    songs_and_urls = get_all_lyrics_urls(top_songs)
    all_lyrics = get_all_lyrics(songs_and_urls)
    if save:
        save_lyrics(all_lyrics)
    print('done scraping.')
    return all_lyrics
