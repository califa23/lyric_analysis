import scraper
import text_normalizer as tn

from nrclex import NRCLex
import matplotlib.pyplot as plt
import nltk
from transformers import pipeline
import spacy
import pytextrank
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

def most_common(normalized_corpus):
    # most common words across all songs
    normalized_corpus = nltk.word_tokenize(normalized_corpus)
    from collections import Counter
    c = Counter(normalized_corpus)
    print('Unique words: ', len(c))
    most_frequent = c.most_common(300)

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

def plot_NRCLex_emotion(data):
    # Create a bar chart
    labels = list(data.keys())
    values = list(data.values())
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    # Add labels and title
    plt.xlabel('Emotion')
    plt.ylabel('Score')
    plt.title('Emotion Scores')
    # Show the plot
    plt.show()

def individual_analysis(normalized_corpus):
    for doc in normalized_corpus:
        # # NRCLex analysis
        # emotion = NRCLex(doc)
        # data = emotion.affect_frequencies
        # print(data)
        # # plot_NRCLex_emotion(data)
        # # most_common(doc)


        # # transformers analysis
        # sentiment_analyzer = pipeline("sentiment-analysis")
        # result = sentiment_analyzer(doc)
        # print(result)


        # # spaCy summarization
        # nlp = spacy.load("en_core_web_sm")
        # nlp.add_pipe("textrank")
        # doc = nlp(doc)
        #
        # top_phrases = sorted(doc._.phrases, key=lambda x: x.rank, reverse=True)[:10]
        # phrases = [phrase.text for phrase in top_phrases]
        # ranks = [phrase.rank for phrase in top_phrases]
        # counts = [phrase.count for phrase in top_phrases]
        #
        # # Plot the results
        # fig, ax = plt.subplots(figsize=(10, 6))
        # ax.barh(phrases, ranks, color='skyblue')
        # ax.set_xlabel('Rank')
        # ax.set_title('Top-Ranked Phrases')
        #
        # for i, count in enumerate(counts):
        #     ax.text(ranks[i] + 0.01, i, f'Count: {count}', va='center', color='black')
        # plt.show()
        # # Extract and print the top-ranked phrases as the summary
        # summary_phrases = [phrase.text for phrase in doc._.phrases if phrase.rank > 0.1]
        # summary = " ".join(summary_phrases)
        #
        # print(summary)
        print('')








def overall_analysis(corpus):
    corpus_concat = ' '.join(corpus)
    noramalized_corpus_concat = tn.normalize_corpusV2(corpus_concat, stopword_removal=True, text_lemmatization=True)
    most_common(noramalized_corpus_concat)

    # NRCLex analysis
    emotion = NRCLex(noramalized_corpus_concat)
    data = emotion.affect_frequencies
    print(data)
    plot_NRCLex_emotion(data)

    # transformer analysis
    sentiment_analyzer = pipeline("sentiment-analysis")
    result = sentiment_analyzer(corpus_concat)
    print(result)





def main():
    print('starting program...')

    # save top 20 song lyrics to txt files
    # scraper.web_scrape_lyrics(save=True, amount=1)

    # load corpus from specified directory
    corpus = scraper.load_corpus_from_saved_files('./resources/test/')
    # at this point, corpus is a list of strings
    # each element being the lyrics to one song
    # not cleaned or tokenized

    print('normalizing...')
    normalized_corpus = tn.normalize_corpus(corpus)
    print('done normalizing.')





    # analysis of each individual song's lyrics
    # individual_analysis(normalized_corpus)

    # analysis of all the lyrics of all the songs
    #overall_analysis(corpus)

    # similarity of all songs
    tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
    dt_matrix = tv.fit_transform(normalized_corpus)
    dt_matrix = dt_matrix.toarray()

    vocab = tv.get_feature_names()
    td_matrix = dt_matrix.T
    pd.DataFrame(np.round(td_matrix, 2), index=vocab).head(10)

    similarity_matrix = np.matmul(dt_matrix, dt_matrix.T)
    print(similarity_matrix.shape)
    print(np.round(similarity_matrix, 3))

    import networkx

    similarity_graph = networkx.from_numpy_array(similarity_matrix)
    plt.figure(figsize=(12, 6))
    networkx.draw_networkx(similarity_graph, node_color='lime')
    plt.show()

    print('program done.')



if __name__ == '__main__':
    main()

