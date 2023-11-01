import re
import urllib3
from bs4 import BeautifulSoup
import time
import random

charts_url = 'https://www.billboard.com/charts/hot-100/'

def get_top(amount = 10):
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
            song_name = re.sub('[^A-Za-z0-9.]+', ' ', song_name_html.text).lower()
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

def get_lyrics(url, song_name):
    http = urllib3.PoolManager()
    page = http.request('GET', url)
    soup = BeautifulSoup(page.data, "html.parser")
    raw = soup.text
    start_pattern = re.compile(r'\n{2}"' + song_name.strip() + '"', re.IGNORECASE)
    matches = list(start_pattern.finditer(raw))
    start = matches[-1].end()
    stop = re.search(r'Submit Corrections', raw).start()
    time.sleep(random.randint(5, 15))
    return raw[start:stop]

def main():
    print('starting...')
    top = get_top(1)
    songs_and_urls = get_all_lyrics_urls(top)
    print(songs_and_urls)
    # for song in songs_and_urls:
    #     print('==========='+song[0]+'===========')
    #     print(get_lyrics(song[2], song[0]).strip())





if __name__ == '__main__':
    main()

