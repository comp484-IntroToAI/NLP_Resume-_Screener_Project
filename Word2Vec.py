# Python program to generate word vectors using Word2Vec

import pdftotext
import os
import itertools
import math

# importing all necessary modules
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim.downloader
import warnings
import os
import numpy as np

warnings.filterwarnings(action = 'ignore')

import gensim
from gensim.models import Word2Vec

# List of stop words
stopWordsList = ["the", "a", "about", "above", "actually", "after", "again", "against", "all", "almost", "also", "although", "always", "am", "an", "and", "any", "are", "as", "at",
        "be", "became", "become", "because", "bin", "before", "being", "below", "between", "both", "but", "by", 
        "can", "could", 
        "did", "do", "does", "doing", "down", "during", 
        "each", "either", "else", 
        "few", "for", "from", "further"
        , "had", "has", "have", "having", "he", "he'd", "he'll", "hence", "he's", "her", "here", "hears", "hers", "herself", "him", "himself", "his", "how", "how's",
        "I", "I'd", "I'll", "I'm", "I've", "if", "if", "in", "into", "is", "it", "it's", "its", "itself"
        "just",
        "let's",
        "may", "maybe", "me", "might", "mine", "more", "most", "must", "my", "myself",
        "neither", "nor", "not", 
        "of", "oh", "on", "once", "only", "okay", "or,", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
        "same", "she", "she'd", "she'll", "she's", "so", "some", "such", 
        "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've'", "this", "those", "through", "to", "too",
        "under", "until", "up",
        "very",
        "was", "we'd", "we", "we'll", "we're", "we've", "were", "what", "what's", "when", "whenever", "when's", "where", "whereas", "wherever", "where's", "whether", "which", "while", "who", "whoever", "who's", "whose", "why", "whom", "why's", "will", "with", "within", "would",
        "yes", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourself", "yourselves"]


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

sampleJob = "NLP Resume Files/Jobs/Sample Job Description.pdf"
job2Text = open("NLP Resume Files/Jobs/Job-Description2.txt")
sampleResume = open("NLP Resume Files/Fullstack-Developer-Resume.txt")
sampleResume = sampleResume.read()

sampleJob2 = job2Text.read()
cleanJob2 = sampleJob2.replace("\n", " ")


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

#=---------MAIN-----------

glove_vectors = gensim.downloader.load("glove-wiki-gigaword-50")
resumeVectorContainer = dict()
resumeList = openResumes(directoryInfoTech)

for i in range(0, len(resumeList), 1):
    original_array = np.empty_like(glove_vectors["test"])
    for word in resumeList[i]: #resumes are string //[] it should be bracketed?
        original_array = original_array + glove_vectors.__getitem__(word)
        # finalVector = vector / resume.__sizeof__()
        # resumeVectorContainer.update({finalVector : resume})

# # Print results
# print(resumeCBOWModel.wv.similarity(openResumes(readSampleResume), getPDFJobDescription(sampleJob)))

# print(resumeVectorContainer)
print("now I can finally rest")

