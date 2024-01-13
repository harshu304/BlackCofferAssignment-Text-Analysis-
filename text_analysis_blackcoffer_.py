
import numpy as np
import re
import os
import nltk
nltk.download('punkt')
import pandas as pd
from nltk.tokenize import RegexpTokenizer , sent_tokenize
from urllib.request import urlopen
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import requests
import urllib.request,sys,time ,requests



stopWordsFile =     'C:\\Users\\harsh\\OneDrive\\Desktop\\Harsh\\StopWords_Generic.txt'
positiveWordsFile = 'C:\\Users\\harsh\\OneDrive\\Desktop\Harsh\\positive-words.txt'
nagitiveWordsFile = 'C:\\Users\\harsh\\OneDrive\\Desktop\\Harsh\\negative-words.txt'

excel_file_path="C:\\Users\\harsh\\OneDrive\\Desktop\\Harsh\\Input.xlsx"
input_file = pd.read_excel(excel_file_path)


def get_article_names(urls):
  titles = []
  for i in range (len(urls)):
    title = urls[i]
    title_clean = title[title.index( "m/" ) + 2 :-1]. replace('-' , ' ')
    titles.append(title_clean)
  return titles

urls =input_file["URL"]
urlsTitleDF = get_article_names(urls)


url = "https://insights.blackcoffer.com/how-people-diverted-to-telehealth-services-and-telemedicine"

page=requests.get(url , headers={"User-Agent": "XY"})
soup = BeautifulSoup(page.text , 'html.parser')
#get title
title = soup . find("h1",attrs = { 'class' : 'entry-title'}).get_text()

#get article text
text = soup . find(attrs = { 'class' : 'td-post-content'}).get_text()
# break into lines and remove leading and trailing space on each
lines = (line.strip() for line in text.splitlines())
# break multi-headlines into a line each
chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
# drop blank lines
text = '\n'.join(chunk for chunk in chunks if chunk)

# Loading positive words
with open(positiveWordsFile,'r') as posfile:
    positivewords=posfile.read().lower()
positiveWordList=positivewords.split('\n')


# Loading negative words
with open(nagitiveWordsFile ,'r' ,  encoding="ISO-8859-1") as negfile:
    negativeword=negfile.read().lower()
negativeWordList=negativeword.split('\n')

#Loading stop words dictionary for removing stop words

with open(stopWordsFile ,'r') as stop_words:
    stopWords = stop_words.read().lower()
stopWordList = stopWords.split('\n')
stopWordList[-1:] = []



print("Positive Words:", positiveWordList[:6])
print("Negative Words:", negativeWordList[:6])
print("Stop Words:", stopWordList[:6])

#tokenizeing module and filtering tokens using stop words list, removing punctuations
def tokenizer(text):
    text = text.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    filtered_words = list(filter(lambda token: token not in stopWordList, tokens))
    return filtered_words

def positive_score (text):
  posword=0
  tokenphrase = tokenizer(text)
  for word in tokenphrase :
    if word in positiveWordList:
       posword+=1

    retpos = posword
    return retpos

def subjectivity_score(text):
    pos_score = positive_score(text)
    neg_score = negative_score(text)
    return (pos_score + neg_score) / total_word_count(text)

def syllable_per_word(text):
    tokens = tokenizer(text)
    total_syllables = 0

    for word in tokens:
        vowels = 0
        for char in word:
            if char in ["a", "e", "i", "o", "u"]:
                vowels += 1

        # Adjust for words ending with 'es' or 'ed'
        if word.endswith(('es', 'ed')):
            vowels -= 1

        total_syllables += max(1, vowels)

    if len(tokens) != 0:
        return total_syllables / len(tokens)
    else:
        return 0



def negative_score (text):
  negword=0
  tokenphrase = tokenizer(text)
  for word in tokenphrase :
    if word in negativeWordList : negword +=1

    retneg = negword
    return retneg

def polarity_score (positive_score , negative_score) :
  return (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
#################################################
def total_word_count(text):
    tokens = tokenizer(text)
    return len(tokens)
#############################################
def AverageSentenceLenght (text):
  Wordcount = len(tokenizer (text))
  SentenceCount = len (sent_tokenize(text))
  if SentenceCount > 0 : Average_Sentence_Lenght = Wordcount / SentenceCount

  avg = Average_Sentence_Lenght

  return round(avg)


# Counting complex words
def complex_word_count(text):
    tokens = tokenizer(text)
    complexWord = 0

    for word in tokens:
        vowels=0
        if word.endswith(('es','ed')):
            pass
        else:
            for w in word:
                if(w=='a' or w=='e' or w=='i' or w=='o' or w=='u'):
                    vowels += 1
            if(vowels > 2):
                complexWord += 1
    return complexWord

def percentage_complex_word(text):
    tokens = tokenizer(text)
    complexWord = 0
    complex_word_percentage = 0

    for word in tokens:
        vowels=0
        if word.endswith(('es','ed')):
            pass
        else:
            for w in word:
                if(w=='a' or w=='e' or w=='i' or w=='o' or w=='u'):
                    vowels += 1
            if(vowels > 2):
                complexWord += 1
    if len(tokens) != 0:
        complex_word_percentage = complexWord/len(tokens)

    return complex_word_percentage

def fog_index(averageSentenceLength, percentageComplexWord):
    fogIndex = 0.4 * (averageSentenceLength + percentageComplexWord)
    return fogIndex

def counts(appended_list, Positive_list, Negative_list, StW, Personal_Pro):
    vowels = ["a", "e", "i", "o", "u"]
    sentence = "".join(map(str, appended_list)).lower()

    # Calculating number of personal pronouns.
    No_of_Personal_Pro = len([i for i in list(sentence.split(" ")) if i in Personal_Pro])

    # Syllable_count
    Syllable_Count = 0
    for words in list(sentence):
        Syllable_Count += len([alphabet for alphabet in words if alphabet in vowels])

def fog_index(text):
    avg_sentence_length = AverageSentenceLenght(text)
    percent_complex_words = percentage_complex_word(text)
    fog_index_value = 0.4 * (avg_sentence_length + percent_complex_words)
    return fog_index_value

def word_count(text):
    words = text.split()
    return len(words)

def personal_pronouns_count(text):
    pronouns = ['I', 'me', 'my', 'mine', 'myself', 'you', 'your', 'yours', 'yourself', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'we', 'us', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourselves', 'they', 'them', 'their', 'theirs', 'themselves']
    count = sum(1 for word in text.split() if word.lower() in pronouns)
    return count

def avg_word_length(text):
    words = text.split()
    if len(words) == 0:
        return 0
    return sum(len(word) for word in words) / len(words)


def avg_words_per_sentence(text):
    word_count = total_word_count(text)
    sentence_count = len(sent_tokenize(text))

    if sentence_count > 0:
        avg_words_per_sentence_value = word_count / sentence_count
    else:
        avg_words_per_sentence_value = 0

    return round(avg_words_per_sentence_value)

URLS = input_file["URL"]

import requests
from bs4 import BeautifulSoup
import time

corps = []
failed_urls = []

for url in URLS:
    try:
        page = requests.get(url, headers={"User-Agent": "XY"})
        page.raise_for_status()  # Raise HTTPError for bad responses

        soup = BeautifulSoup(page.text, 'html.parser')

        # get title
        title = soup.find("h1", attrs={'class': 'entry-title'}).get_text()

        # get article text
        text = soup.find(attrs={'class': 'td-post-content'}).get_text()

        # process and clean text (if needed)
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        corps.append(text)
        time.sleep(1)

    except requests.exceptions.RequestException as e:
        print(f"Error for URL {url}: {e}")
        failed_urls.append(url)

    except AttributeError:
      page = requests.get(url, headers={"User-Agent": "XY"})
      #page.raise_for_status()  # Raise HTTPError for bad responses

      soup = BeautifulSoup(page.text, 'html.parser')

        # get title
      title = soup.find("h1", attrs={'class': 'tdb-title-text'}).get_text()

        # get article text
      text = soup.find(attrs={'class': 'tdc-content-wrap'}).get_text()

        # process and clean text (if needed)
      lines = (line.strip() for line in text.splitlines())
      chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
      text = '\n'.join(chunk for chunk in chunks if chunk)

      corps.append(text)
      time.sleep(1)

def delete_rows_from_excel(excel_file_path, urls_to_delete):
    try:
        # Read the Excel file into a DataFrame
        df = pd.read_excel(excel_file_path)

        # Filter rows based on URLs to delete
        df = df[~df['URL'].isin(urls_to_delete)]

        # Save the updated DataFrame back to the Excel file
        df.to_excel(excel_file_path, index=False)

        print("Rows deleted successfully.")
    except Exception as e:
        print(f"Error: {e}")

  # Add the URLs you want to delete


print("Length of 'corps' list:", len(corps))
print("Failed URLs:", failed_urls)
delete_rows_from_excel(excel_file_path, failed_urls)

excel_file_path="C:\\Users\\harsh\\OneDrive\\Desktop\\Harsh\\Input.xlsx"
input_file = pd.read_excel(excel_file_path)
print(len(urlsTitleDF))
print(len(corps))



df = pd.DataFrame({'title':urlsTitleDF,'corps': corps})
df["total word count"] = df["corps"] . apply (total_word_count)
df["positive_score"] = df["corps"] . apply (positive_score)
df["negative_score"] = df["corps"] . apply (negative_score)
df["polarity_score"] = np.vectorize(polarity_score)(df['positive_score'],df['negative_score'])
df["SUBJECTIVITY SCORE"] = df["corps"] . apply (subjectivity_score)
df["AverageSentenceLenght"] = df["corps"] . apply (AverageSentenceLenght)
df["percentage_complex_word"] = df["corps"] . apply (percentage_complex_word)
df["FOG INDEX"] = df["corps"] . apply (fog_index)
df["AVG NUMBER OF WORDS PER SENTENCE"] = df["corps"] . apply (avg_words_per_sentence)
df["complex_word_count"] = df["corps"] . apply (complex_word_count)
df["word_count"] = df["corps"] . apply (word_count)
df["SYLLABLE PER WORD"] = df["corps"] . apply (syllable_per_word)
df["PERSONAL PRONOUNS"] = df["corps"] . apply (syllable_per_word)
df["AVG WORD LENGTH"] = df["corps"] . apply (syllable_per_word)







final = df.drop("corps", axis=1)


#df.index = range(1, len(df) + 1)
df.index = df.index + 1
final.to_excel('Output Data Structure.xlsx')



