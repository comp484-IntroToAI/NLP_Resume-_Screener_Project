# Python program to generate word vectors using Word2Vec
# !pip install pdftotext
# import pdftotext
import os
import itertools
import math
import re

# importing all necessary modules
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim.downloader
import warnings
import os
import numpy as np

warnings.filterwarnings(action = 'ignore')

sampleResume = open("/content/Edited ShomrikMondal_Resume.txt")

# Isaac's Drive Pathway
# sampleResume = open("/content/drive/MyDrive/Intro to A.I./A.I. NLP Project/NLP A.I. Resume Files/Test Resumes TXT/AlbertLuna_Resume.txt")

job2Text = open("/content/Job-Description2.txt")
sampleText = sampleResume.read()
jobText = job2Text.read()
sampleResume.close()
job2Text.close()
# print(jobText)
def cleanText(txtFile):
    # Removes any special characters from inside the word and strips it to its basic
    txtFile.lower()
    txtFile.strip()
    newStr = ""
    for word in txtFile.split():
      to_array = [char for char in word]
      count = 0
      for char in to_array:
        if (ord(char) < 65) or (96 > ord(char) > 90) or (122 < ord(char)):
          to_array[count] = " "
        count = count + 1
      newStr += convert(to_array)
    return newStr

def convert(s):

    # initialization of string to ""
    new = ""

    # traverse in the string
    for x in s:
        new += x
    new += " "
    # return string
    return new

from sklearn.metrics.pairwise import cosine_similarity
import torch as torch
import torch.nn.functional as Y


newText = cleanText(sampleText)

newText2 = cleanText(jobText)
finalVector = np.empty([2, 300])

vocab = set(newText.split())
vocab2 = set(newText2.split())
vocab = vocab.union(set(vocab2))
vocab = list(vocab)
# print(vocab)



vA = np.zeros(len(vocab), dtype=float)
vB = np.zeros(len(vocab), dtype=float)
for w in newText.split():
  try:
    i = vocab.index(w)
    vA[i] += 1
  except ValueError:
    pass

for w in newText2.split():
  try:
    i = vocab.index(w)
    vB[i] += 1
  except ValueError:
    pass
# print(vA)
# print(vB)
cos = np.dot(vA, vB) / (np.sqrt(np.dot(vA,vA)) * np.sqrt(np.dot(vB,vB)))
print(cos)


