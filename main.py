import re
import urllib3
from bs4 import BeautifulSoup
import time
import random
import os

charts_url = 'https://www.billboard.com/charts/hot-100/'

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
            song_name = re.sub(' +', ' ', song_name_html.text).lower()
            artist = re.sub('[^A-Za-z0-9.&]+', ' ', artist_html.text).lower()
            artist = artist.split('&')[0]
            artist = artist.split('featuring')[0]
            top.append((song_name, artist))
            count += 1
    return top

def get_lyrics_url(song_name, artist):
    song_name = re.sub('[^A-Za-z0-9.]+','', song_name).lower()
    artist = re.sub('[^A-Za-z0-9.]+','', artist).lower()
    return 'https://www.azlyrics.com/lyrics/' + artist + '/' + song_name + '.html'

def get_all_lyrics_urls(songs):
    urls = []
    for song in songs:
        urls.append((song[0], song[1], get_lyrics_url(song[0], song[1])))
    return urls

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

def save_all_lyrics(songs_and_urls):
    print('saving lyrics to resources...')
    count = 0
    for song in songs_and_urls:
        count += 1
        print('--' + str(round((count/len(songs_and_urls))*100,2)) + '%: ' + song[0].strip(), end='')
        f = open('./resources/' + song[0].strip() + '.txt', 'w+')
        f.write(get_lyrics(song[2], song[0], song[1]).strip())
        print('--' + str(f.tell()))
        f.close()
    print('done saving.')

def load_corpus_from_saved_files():
    print('creating corpus from resources...')
    corpus = ""
    file_names = [f for f in os.listdir('./resources') if os.path.isfile(os.path.join('./resources', f))]
    for file_name in file_names:
        file = open('./resources/' + file_name, 'r')
        for line in file:
            corpus += line
        file.close()
    print('done creating corpus.')
    return corpus

def web_scrape_corpus(save = False):
    print('scraping for corpus...')
    corpus = ""
    count = 0
    top_songs = get_top_songs(20)
    songs_and_urls = get_all_lyrics_urls(top_songs)
    for song in songs_and_urls:
        count += 1
        print('--' + str(round((count/len(songs_and_urls))*100,2)) + '%: ' + song[0].strip(), end='')
        corpus += get_lyrics(song[2], song[0], song[1]).strip()
    print('done scraping.')
    if save:
        save_all_lyrics(songs_and_urls)
    return corpus

def main():
    print('starting program...')
    have_saved = True
    if have_saved:
        corpus = load_corpus_from_saved_files()
    else:
        corpus = web_scrape_corpus(False)

    

    print('program done.')



if __name__ == '__main__':
    main()

