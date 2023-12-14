# Python program to generate word vectors using Word2Vec

import pdftotext
import os
import itertools
import math
import re
import torch as torch
import torch.nn.functional as Y



# importing all necessary modules
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim.downloader
import warnings
import os
import numpy as np

warnings.filterwarnings(action = 'ignore')

import gensim
from gensim.models import Word2Vec
from gensim.test.utils import datapath

directoryAccountant = 'NLP Resume Files/data/ACCOUNTANT/'
directoryAdvocate = 'NLP Resume Files/data/ADVOCATE/'
directoryAgriculture = 'NLP Resume Files/data/AGRICULTURE/'
directoryApparel = 'NLP Resume Files/data/APPAREL/'
directoryArts = 'NLP Resume Files/data/ARTS/'
directoryAutomobile = 'NLP Resume Files/data/AUTOMOBILE/'
directoryAviation = 'NLP Resume Files/data/AVIATION/'
directoryBanking = 'NLP Resume Files/data/BANKING/'
directoryBpo = 'NLP Resume Files/data/BPO/'
directoryBusiness = 'NLP Resume Files/data/BUSINESS-DEVELOPMENT/'
directoryChef = 'NLP Resume Files/data/CHEF/'
directoryConstruction = 'NLP Resume Files/CONSTRUCTION/'
directoryConsultant = 'NLP Resume Files/data/CONSULTANT/'
directoryDesigner = 'NLP Resume Files/data/DESIGNER/'
directoryDigitalMedia = 'NLP Resume Files/data/DIGITAL-MEDIA/'
directoryEngineering = 'NLP Resume Files/data/ENGINEERING/'
directoryFinance = 'NLP Resume Files/data/FINANCE/'
directoryFitness = 'NLP Resume Files/data/FITNESS/'
directoryHealthcare = 'NLP Resume Files/data/HEALTHCARE/'
directoryHR = 'NLP Resume Files/data/HR/'
directoryInfoTech = 'NLP Resume Files/data/INFORMATION-TECHNOLOGY/'
directoryPR = 'NLP Resume Files/data/PUBLIC-RELATIONS/'
directorySales = 'NLP Resume Files/data/SALES/'
directoryTeacher = 'NLP Resume Files/data/TEACHER/'

job2Text = open("/Users/jackkeller/Desktop/484F23/project-jack_isaac/NLP Resume Files/Jobs/Job-Description2.txt")

sampleJob2 = job2Text.read()

jobDescriptionData = []

# Create CBOW model (Continuous bag of words)

#resumeCBOWModel = Word2Vec(createTokenFromText(openResumes(directoryInfoTech)), min_count = 1, vector_size = 100, window = 5)

def openResumes(resumeDirectory):
    resumeList = []
    for pdfFile in os.listdir(resumeDirectory):
        completeResume = os.path.join(resumeDirectory, pdfFile)
        if os.path.isfile(completeResume):
            with open (completeResume, "rb") as f:
                pdf = pdftotext.PDF(f)
                resumeText = "\n\n".join(pdf)
                cleanResumeText = resumeText.replace("\n", " ")
                resumeList.append(cleanResumeText)
                return resumeList

def getPDFJobDescription(jobPath):
    with open (jobPath, "rb") as j:
        jobText = pdftotext.PDF(j)
        completejobDesciption = "\n\n".join(jobText)
    return completejobDesciption

# iterate through each sentence in the file
def createTokenFromText(textInput):
      resumeData = []
      for i in sent_tokenize(textInput):
            temp = []
            for j in word_tokenize(i):
                temp.append(j.lower())
                resumeData.append(temp)
            return resumeData


def cleanText(txtFile):
    # Removes any special characters from inside the word and strips it to its basic
    txtFile.lower()
    txtFile.strip()
    newStr = ""
    for word in txtFile.split():
      to_array = [char for char in word]
      count = 0
      for char in to_array:
        print(char)
        print(ord(char))
        if (ord(char) < 65) or (96 > ord(char) > 90) or (122 < ord(char)):
          print("OMG")
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

#=---------MAIN-----------

glove_vectors = gensim.downloader.load("glove-wiki-gigaword-300")
glove_vectors.most_similar('twitter')
resumeVectorContainer = dict()
resumeDirectory = openResumes(directoryInfoTech)

string = "as;dflkj!@#$%^-_"
# print(cleanText(string))
jobText = cleanText(sampleJob2)

resumeScoreHolder = dict()
for file in os.listdir(directoryInfoTech):
    completeResume = os.path.join(resumeDirectory, file)
    if os.path.isfile(completeResume):
        with open (completeResume, "r") as singleResume:
            readResume = singleResume.read()
            # pdf = pdftotext.PDF(f)
            # resumeText = "\n\n".join(pdf)
            finalResumeV = np.empty([2, 300])
            for word in readResume.split():
                try:
                    vector = glove_vectors[word]
                    finalResumeV += vector
                except KeyError:
                    pass
                except NameError:
                    pass
            centroidResume = finalResumeV / len(readResume.split())
            finalJobV = np.empty([2, 300])
            for word in jobText.split():
                try:
                    vector = glove_vectors[word]
                    finalJobV += vector
                except KeyError:
                    pass
                except NameError:
                    pass
            centroidJob = finalJobV / len(jobText.split())
            updatedArray = np.reshape(np.array([centroidJob]), (300, 2))
            tensorJob = torch.from_numpy(centroidJob)
            tensorResume = torch.from_numpy(centroidResume)
            print(Y.cosine_similarity(tensorJob, tensorResume))
            print('------ new resume  ------')
            # result = str(similarity_matrix[1][0]*100)
            # print('Current Resume:' + file)
            # print('Resume matches by:'+ result + '%\n')
            # resumeScoreHolder.update({completeResume : result})

    # print(resumeVectorContainer)
    print("now I can finally rest")

