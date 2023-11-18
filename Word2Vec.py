# Python program to generate word vectors using Word2Vec

# importing all necessary modules
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
import os

warnings.filterwarnings(action = 'ignore')

import gensim
from gensim.models import Word2Vec

# Reads ‘alice.txt’ file
sample = open("C:\\Users\\Admin\\Desktop\\alice.txt", "utf8")
s = sample.read()

def openResumes(resumeDirectory)
    for pdfFile in os.listdir(resumeDirectory):
        completeResume = os.path.join(resumeDirectory, pdfFile)
        if os.path.isfile(completeResume):
            with open (completeResume, "rb") as f:
                pdf = pdftotext.PDF(f)
                resumeText = "\n\n".join(pdf)


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

# Replaces escape character with space
f = s.replace("\n", " ")

data = []

# iterate through each sentence in the file
for i in sent_tokenize(f):
	temp = []
	
	# tokenize the sentence into words
	for j in word_tokenize(i):
		temp.append(j.lower())

	data.append(temp)

# Create CBOW model
model1 = gensim.models.Word2Vec(data, min_count = 1, 
							vector_size = 100, window = 5)

# Print results
print("Cosine similarity between 'alice' " +
			"and 'wonderland' - CBOW : ",
	model1.wv.similarity('alice', 'wonderland'))
	
print("Cosine similarity between 'alice' " +
				"and 'machines' - CBOW : ",
	model1.wv.similarity('alice', 'machines'))

# Create Skip Gram model
model2 = gensim.models.Word2Vec(data, min_count = 1, vector_size = 100,
											window = 5, sg = 1)

# Print results
print("Cosine similarity between 'alice' " +
		"and 'wonderland' - Skip Gram : ",
	model2.wv.similarity('alice', 'wonderland'))
	
print("Cosine similarity between 'alice' " +
			"and 'machines' - Skip Gram : ",
	model2.wv.similarity('alice', 'machines'))
