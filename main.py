import scraper
import text_normalizer as tn

def main():
    print('starting program...')

    have_saved = False
    if have_saved:
        corpus = scraper.load_corpus_from_saved_files()
    else:
        corpus = scraper.web_scrape_corpus(save=False, amount=1)

    print('normalizing...')
    normalized_lyrics = tn.normalize_corpus(corpus)
    print('done normalizing.')
    print(normalized_lyrics[:200])

    print('program done.')



if __name__ == '__main__':
    main()

