import scraper
import text_normalizer as tn

import os
from datetime import date
import matplotlib.pyplot as plt
from nrclex import NRCLex
import nltk
from transformers import pipeline
import spacy
import pytextrank
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import networkx as nx

def most_common(song_name, doc, plot = False):
    # most common words across all songs
    normalized_corpus = nltk.word_tokenize(doc)
    from collections import Counter
    c = Counter(normalized_corpus)
    print('Unique words: ', len(c))
    most_frequent = c.most_common(300)
    print(most_frequent[0])

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
    plt.title(song_name)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Remove axis

    path = './plots/' + str(date.today()) + '/mostCommon/'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + song_name)
    plt.close()

def plot_NRCLex_emotion(data, title):
    labels = list(data.keys())
    values = list(data.values())
    plt.bar(labels, values)
    plt.xticks(rotation=30, ha="right")
    plt.xlabel('Emotion')
    plt.ylabel('Score')
    plt.title(title)

    path ='./plots/' + str(date.today()) + '/emotions/'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + title)
    plt.close()

def NRCLex_analysis(song_name, doc, plot = False):
    emotion = NRCLex(doc)
    data = emotion.affect_frequencies
    print(song_name + ': ' + max(data, key=data.get))

    if plot:
        plot_NRCLex_emotion(data, song_name)

def individual_analysis(normalized_corpus):
    # song_names, song_lyrics = zip(*normalized_corpus)
    for song_name, doc in normalized_corpus:
        # NRCLex analysis
        NRCLex_analysis(song_name, doc, True)
        most_common(song_name, doc, True)


        # transformers analysis
        # sentiment_analyzer = pipeline("sentiment-analysis")
        # result = sentiment_analyzer(doc)
        # print(result)


        # spaCy summarization
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
        # print(summary)

def overall_analysis(corpus):
    song_names, song_lyrics = zip(*corpus)
    corpus = song_lyrics
    corpus_concat = ' '.join(corpus)
    noramalized_corpus_concat = tn.normalize_corpusV2(corpus_concat, stopword_removal=True, text_lemmatization=True)

    most_common('All Songs', noramalized_corpus_concat, True)
    plot_NRCLex_emotion(noramalized_corpus_concat, 'All Songs')

    # # transformer analysis
    # sentiment_analyzer = pipeline("sentiment-analysis")
    # result = sentiment_analyzer(corpus_concat)
    # print(result)

def tfidf_similarity(normalized_corpus, song_index_to_compare):
    song_names, song_lyrics = zip(*normalized_corpus)

    tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
    dt_matrix = tv.fit_transform(song_lyrics)
    dt_matrix = dt_matrix.toarray()
    similarity_vector = np.dot(dt_matrix, dt_matrix[song_index_to_compare])

    # Print the similarity scores
    similarity_scores = pd.DataFrame({'Song': song_names, 'Similarity': similarity_vector})
    print(similarity_scores)

    # Create a similarity graph
    similarity_graph = nx.Graph()
    for i, (name, score) in enumerate(zip(song_names, similarity_vector)):
        similarity_graph.add_node(i, label=name)
        similarity_graph.add_edge(song_index_to_compare, i, weight=score)

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(similarity_graph)
    labels = nx.get_node_attributes(similarity_graph, 'label')
    edge_labels = nx.get_edge_attributes(similarity_graph, 'weight')
    nx.draw_networkx(similarity_graph, pos, node_color='lime', labels=labels, with_labels=True)
    nx.draw_networkx_edge_labels(similarity_graph, pos, edge_labels=edge_labels, font_color='red')
    plt.title(f"Similarity of {song_names[song_index_to_compare]} to Other Songs")
    plt.show()

def clustering(normalized_corpus):
    song_names, song_lyrics = zip(*normalized_corpus)
    stop_words = nltk.corpus.stopwords.words('english')
    cv = CountVectorizer(ngram_range=(1, 4), max_df=4, stop_words=stop_words)
    cv_matrix = cv.fit_transform(song_lyrics)
    print(cv_matrix.shape)

    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=0, n_iter=10000, perplexity=2)
    np.set_printoptions(suppress=True)
    T = tsne.fit_transform(cv_matrix)

    plt.figure(figsize=(14, 8))
    plt.scatter(T[:, 0], T[:, 1], c='red', edgecolors='k')
    for label, x, y in zip(song_names, T[:, 0], T[:, 1]):
        plt.annotate(label, xy=(x + 1, y + 1), xytext=(0, 0), textcoords='offset points')

    plt.show()


def main():
    print('starting program...')

    # save top 20 song lyrics to txt files
    # scraper.web_scrape_lyrics(save=True, amount=30)

    # load corpus from specified directory
    corpus = scraper.load_corpus_from_saved_files('./resources/test2/')
    # at this point, corpus is a list of stringse
    # each element being the lyrics to one song
    # not cleaned or tokenized

    print('normalizing...')
    normalized_corpus = tn.normalize_corpus(corpus)
    print('done normalizing.')


    print('starting analysis...\n\n\n')
    # analysis of each individual song's lyrics in corpus
    individual_analysis(normalized_corpus)

    # analysis of all the lyrics of all the songs (use a different form of normalization)
    overall_analysis(corpus)

    # Tfidf similarity (of whole corpus)
    # tfidf_similarity(normalized_corpus, 0)

    # Clustering (of whole corpus)
    # clustering(normalized_corpus)

    print('program done.')



if __name__ == '__main__':
    main()

