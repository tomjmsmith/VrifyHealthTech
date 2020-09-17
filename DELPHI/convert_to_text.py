
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as SIA
import nltk
import io
import os
from nltk.corpus import stopwords


# set the stopwords to be the english version
stop_words = set(stopwords.words("english"))

# vader sentiment analyzer for analyzing the sentiment of the text
sid = SIA()

# function for analyzer score


def sentiment_analyzer_scores(sentence):
    score = sid.polarity_scores(sentence)


def write_to_file(f, f1):  # this writes and filters the words
    with open("../Speech_to_text_test/my_result.txt") as f:
        with open("../Speech_to_text_test/clean_text.txt", "w+") as f1:
             # for every line in f, we split up the words,
             # filter out stop words, give it a part of speech,
             # and then write it to the new file
            for line in f:
                ss = sid.polarity_scores(line) # here we get the polarity
                new = line.split() # here we split the lines, this inherently tokenizes the words
                filtered_words = [words for words in new if not words in stop_words] # this filters out the stopwords
                tagged = nltk.pos_tag(filtered_words) # this prints the part of speech
                f1.write(str(tagged)) # this writes the token word with part of speech
                f1.write(str(ss)) # polarity score (i.e. positive, negative, neutral)
                f1.write("\n") # new line
                namedEnt = nltk.ne_chunk(tagged) # chunks all the ish

            f1.close() # closes the new file
        f.close() # closes the old file