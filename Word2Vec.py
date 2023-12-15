# Python program to generate word vectors using Word2Vec
import pdftotext
import torch as torch
import torch.nn.functional as Y
from sklearn.metrics.pairwise import cosine_similarity

# Python program to generate word vectors using Word2Vec
# import pdftotext
import os

# importing all necessary modules
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
import gensim.downloader
import warnings
import os
import numpy as np

from gensim.test.utils import datapath

# importing all necessary modules
warnings.filterwarnings(action = 'ignore')

from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.test.utils import datapath

# glove_model = KeyedVectors.load_word2vec_format('content/glove.6B.300d.txt', binary=False)

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
directoryInfoTech = '/Users/jackkeller/Desktop/484F23/project-jack_isaac/NLP Resume Files/data/INFORMATION-TECHNOLOGY'
directoryPR = 'NLP Resume Files/data/PUBLIC-RELATIONS/'
directorySales = 'NLP Resume Files/data/SALES/'
directoryTeacher = 'NLP Resume Files/data/TEACHER/'

job2Text = open("NLP Resume Files/Jobs/Test Sample.txt")
sampleResume = open("NLP Resume Files/Jobs/Test Sample.txt")

sampleText = sampleResume.read()
jobText = job2Text.read()
sampleResume.close()
job2Text.close()

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

def load_glove_model(File):
    print("Loading Glove Model")
    glove_model = {}
    with open(File,'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    return glove_model

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
    txtFile = txtFile.lower()
    txtFile = txtFile.strip()
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

#=---------MAIN-----------

glove_vectors = gensim.downloader.load("glove-wiki-gigaword-300")
# model = KeyedVectors.load_word2vec_format('/root/gensim-data/glove-wiki-gigaword-300/glove-wiki-gigaword-300.gz')

# gloveNPY = 'content/glove-wiki-gigaword-300.model.vectors.npy'
# gloveTXT = 'content/glove.6B.300d.txt'


# print(load_glove_model(gloveTXT))

# glove_vectors.most_similar('twitter')
# resumeVectorContainer = dict()
# resumeDirectory = openResumes(directoryInfoTech)

cleanResumeText = cleanText(sampleText)
cleanJobText = cleanText(jobText)
finalResumeVec = np.empty(300)
finalJobVec = np.empty(300)

resumeCounter = 0
jobCounter = 0

resumeScoreHolder = dict()
# for file in os.listdir(directoryInfoTech):
#     completeResume = os.path.join(resumeDirectory, file)
#     if os.path.isfile(completeResume):
#         with open (completeResume, "rb") as f:
#             pdf = pdftotext.PDF(f)
#             resumeText = "\n\n".join(pdf)
#             finalResumeV = np.empty([2, 300])

for word in cleanResumeText.split():
    try:
        vector = glove_vectors[word]
        finalResumeVec += vector
        resumeCounter += 1
    except KeyError:
        pass
    except NameError:
        pass
centroidResume = finalResumeVec.reshape(1, -1) / resumeCounter

for word in cleanJobText.split():
    try:
        vector = glove_vectors[word]
        finalJobVec += vector
        jobCounter += 1
    except KeyError:
        pass
    except NameError:
        pass

centroidJob = finalJobVec.reshape(1, -1) / jobCounter

# updatedArray = np.reshape(np.array([centroidJob]), (300, 2))
tensorResume = torch.from_numpy(centroidResume)
tensorJob = torch.from_numpy(centroidJob)


cosinePair = Y.cosine_similarity(tensorJob, tensorResume)
cosineScore = cosinePair[0]
print(cosineScore.item())
    # if (newVal.item() > 0.96):
    #     counter = counter + 1
    # print(counter)
    # result = str(similarity_matrix[1][0]*100)
    # print('Current Resume:' + file)
    # print('Resume matches by:'+ result + '%\n')
    # resumeScoreHolder.update({completeResume : result})
