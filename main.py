import scraper
import text_normalizer as tn

def main():
    print('starting program...')

    have_saved = True
    if have_saved:
        corpus = scraper.load_corpus_from_saved_files()
    else:
        corpus = scraper.web_scrape_corpus(save=False, amount=1)


    print('normalizing...')
    normalized_lyrics = tn.normalize_corpusV2(corpus)
    print('done normalizing.')

    from collections import Counter
    c = Counter(normalized_lyrics)
    print(len(c))
    most_frequent = c.most_common(300)
    print(most_frequent)




    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    # Extract words and counts
    words = [word for word, count in most_frequent]
    counts = [count for word, count in most_frequent]

    # Create a dictionary with words and their frequencies
    word_freq = {words[i]: counts[i] for i in range(len(words))}

    # Create a WordCloud object
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

    # Plot the WordCloud
    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Remove axis
    plt.show()

    print('program done.')



if __name__ == '__main__':
    main()

