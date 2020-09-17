import speech_recognition as sr
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as SIA
import pyttsx3

# set the stopwords to be the english version
stop_words = set(stopwords.words("english"))

engine = pyttsx3.init()

# vader sentiment analyzer for analyzing the sentiment of the text
sid = SIA()

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
                # namedEnt = nltk.ne_chunk(tagged) # chunks all the ish

            f1.close() # closes the new file
        f.close() # closes the old file


# this is where we gather the data
# create the recognizer
r = sr.Recognizer()
# define the microphone
mic = sr.Microphone(device_index=0)


# use the microphone to record speech
# should this be a class?
with mic as source:
    while True:
        engine.say("What is your name?")
        engine.runAndWait()
        audio_data = r.listen(source)
        print("recognizing")
        r.adjust_for_ambient_noise(source)
        text = r.recognize_google(audio_data)
        print(text.lower())

        with open('../Speech_to_text_test/my_result.txt', mode='a+') as file:
            file.write("Name:")
            file.write(text.lower())
            file.write("\n")
            file.close()
        break

with mic as source:
    while True:
        engine.say("How can I help you today?")
        engine.runAndWait()
        audio_data = r.listen(source)
        r.adjust_for_ambient_noise(source)
        text = r.recognize_google(audio_data)
        print(text.lower())

        with open('../Speech_to_text_test/my_result.txt', mode='a+') as file:
            file.write(text)
            file.write("\n")
            file.close()
        break

with mic as source:
    while True:
        engine.say("""
                    I'm sorry you're feeling that way. This is really normal though, so don't worry. 
                    Would you like to talk with someone?
                    """)
        engine.runAndWait()
        audio_data = r.listen(source)
        r.adjust_for_ambient_noise(source)
        text = r.recognize_google(audio_data)
        print(text.lower())

        with open('../Speech_to_text_test/my_result.txt', mode='a+') as file:
            file.write(text)
            file.write("\n")
            file.close()
        break

with mic as source:
    while True:
        engine.say("Okay. Hold on Tom, we'll get you to someone who is right for you. Where are you located?")
        engine.runAndWait()
        audio_data = r.listen(source)
        r.adjust_for_ambient_noise(source)
        text = r.recognize_google(audio_data)
        print(text.lower())

        with open('../Speech_to_text_test/my_result.txt', mode='a+') as file:
            file.write("Location:")
            file.write(text)
            file.write("\n")
            file.close()
        break

with mic as source:
    while True:
        engine.say("""
                    There are over 200 mental health providers in Northbrook, Illinois. 
                    We recommend you see a Licensed Clinical Social Worker.
                    55 have availability for tomorrow. Do you want to use insurance or pay out of pocket.
                    """)
        engine.runAndWait()
        audio_data = r.listen(source)
        r.adjust_for_ambient_noise(source)
        text = r.recognize_google(audio_data)
        print(text.lower())

        with open('../Speech_to_text_test/my_result.txt', mode='a+') as file:
            file.write(text)
            file.write("\n")
            file.close()
        break

with mic as source:
    while True:
        engine.say("""
            Okay. There are 15 providers available. Do you want us to set up an appointment? 
            Do you have any gender preference? Or time you're available.
             """)
        engine.runAndWait()
        audio_data = r.listen(source)
        r.adjust_for_ambient_noise(source)
        text = r.recognize_google(audio_data)
        print(text.lower())

        with open('../Speech_to_text_test/my_result.txt', mode='a+') as file:
            file.write(text)
            file.write("\n")
            file.close()
        break

with mic as source:
    while True:
        engine.say("""
                Perfect, hang in there. We're setting up an appointment for you with Mr. Michael Smith at 9 AM.
                The first session is free because you used us. Say yes if you'd like us to send over your file.
                """)
        engine.runAndWait()
        audio_data = r.listen(source)
        r.adjust_for_ambient_noise(source)
        text = r.recognize_google(audio_data)
        print(text.lower())

        with open('../Speech_to_text_test/my_result.txt', mode='a+') as file:
            file.write(text)
            file.write("\n")
            file.close()
        break

with mic as source:
    while True:
        engine.say("""Terrific. We just sent it. Now all we need is your phone number.
                            """)
        engine.runAndWait()
        audio_data = r.listen(source)
        r.adjust_for_ambient_noise(source)
        text = r.recognize_google(audio_data)
        print(text.lower())

        with open('../Speech_to_text_test/my_result.txt', mode='a+') as file:
            file.write(text)
            file.write("\n")
            file.close()

        # else:
        #     engine.say("""No problem. Now all we need is your phone number. """)
        #     engine.runAndWait()
        #     audio_data = r.listen(source)
        #     r.adjust_for_ambient_noise(source)
        #     text = r.recognize_google(audio_data)
        #     print(text.lower())
        #
        #     with open('../Speech_to_text_test/my_result.txt', mode='a+') as file:
        #         file.write(text)
        #         file.write("\n")
        #         file.close()

        break

with mic as source:
    while True:
        engine.say("""You're good to go with Michael Smith tomorrow at 9 AM. 
        We'll send you a secure text with his address and a reminder. Do you need help with anything else?""")
        engine.runAndWait()
        audio_data = r.listen(source)
        r.adjust_for_ambient_noise(source)
        text = r.recognize_google(audio_data)
        print(text.lower())

        with open('../Speech_to_text_test/my_result.txt', mode='a+') as file:
            file.write(text)
            file.write("\n")
            file.close()

        break

print("Exporting process completed! Now beginning the NLP.")

write_to_file('../Speech_to_text_test/my_result.txt', '../Speech_to_text_test/clean_text.txt')
