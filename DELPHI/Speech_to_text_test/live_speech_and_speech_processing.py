# import speech recognizer
import speech_recognition as sr
# import nltk
import nltk
import io
from nltk.corpus import stopwords
from nltk.corpus import WordListCorpusReader
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import cv2
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as SIA


# set the stopwords to be the english version
stop_words = set(stopwords.words("english"))

# vader sentiment analyzer for analyzing the sentiment of the text
sid = SIA()

# function for analyzer score
def sentiment_analyzer_scores(sentence):
    score = sid.polarity_scores(sentence)

# Step 2. Design the Vocabulary
# The default token pattern removes tokens of a single character.
# That's why we don't have the "I" and "s" tokens in the output
count_vectorizer = CountVectorizer()

# this is where we gather the data
# create the recognizer
r = sr.Recognizer()
# define the microphone
mic = sr.Microphone(device_index=0)
# invite the user to talk
print("Please talk.")

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

# open cv webcam
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    cv2.imshow("img", img)
    k = cv2.waitKey(30) & 0xff

    if k == 27:
        break

# use the microphone to record speech
# should this be a class?
with mic as source:
    while True:
        audio_data = r.listen(source)
        print("Recognizing...")
        r.adjust_for_ambient_noise(source)
        text = r.recognize_google(audio_data)
        print(text.lower())

# write the text to a file
        with open('my_result.txt', mode='w+') as file:
            file.write("Recognized text:")
            file.write("\n")
            file.write(text)
            file.close()
# dont touch the stuff before this
            if text == "bye":
                break
        continue

print("Exporting process completed! Now beginning the NLP.")


with open("my_result.txt") as f:
    with open("clean_text.txt", "w+") as f1:
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


cap.release()
cv2.destroyAllWindows()

#namedEnt.draw()

# Step 3. Create the Bag-of-Words Model
# bag_of_words = count_vectorizer.fit_transform(documents)
# feature_names = count_vectorizer.get_feature_names()
# pd.DataFrame(bag_of_words.toarray(), columns=feature_names)


