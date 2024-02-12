from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import nltk

# Ensure you have the necessary NLTK data (uncomment if needed)
# nltk.download('punkt')
# nltk.download('stopwords')


def chop_and_filter(s, top_words_count=5):
    """Process input Series to identify and return the top N frequent, significant words,
    filtering based on stopwords and similarity (Jaccard index)."""
    # Tokenize the concatenated string and filter out stopwords and non-alphabetic words in one step
    stop_words = set(stopwords.words("english"))
    words = [
        word.lower()
        for word in word_tokenize(s.str.cat(sep=" "))
        if word.isalpha() and word.lower() not in stop_words
    ]

    # Simplify the word list to remove similar words based on the Jaccard index
    simplified_words = simplify_word_list(words, threshold=0.90)

    # Count the occurrences of each word and sort them by frequency in descending order
    word_counts = Counter(simplified_words)
    sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)

    # Select the top N words
    top_words = sorted_words[:top_words_count]
    return ", ".join(top_words)


def simplify_word_list(words, threshold=0.90):
    """Filter out words that are too similar based on the Jaccard index."""
    filtered_words = []
    for word in words:
        word_set = set(word)
        if all(
            jaccard_index(word_set, set(other_word)) < threshold for other_word in filtered_words
        ):
            filtered_words.append(word)
    return filtered_words


def jaccard_index(set1, set2):
    """Calculate the Jaccard Index of two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union else 0
