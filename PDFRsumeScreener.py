import pdftotext
import os
import itertools
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
def getPDFJobDescription(jobPath):
    with open (jobPath, "rb") as j:
        jobText = pdftotext.PDF(j)
        completejobDesciption = "\n\n".join(jobText)
    return completejobDesciption

def getBestResume(resumeDirectory, jobText):
    resumeScoreHolder = dict()
    for pdfFile in os.listdir(resumeDirectory):
        completeResume = os.path.join(resumeDirectory, pdfFile)
        if os.path.isfile(completeResume):
            with open (completeResume, "rb") as f:
                pdf = pdftotext.PDF(f)
                resumeText = "\n\n".join(pdf)
                content = [jobText, resumeText]
                cv = CountVectorizer()
                matrix = cv.fit_transform(content)
                similarity_matrix = cosine_similarity(matrix)
                print('------ resume  ------')
                result = str(similarity_matrix[1][0]*100)
                print('Current Resume:' + pdfFile)
                print('Resume matches by:'+ result + '%\n')
                resumeScoreHolder.update({completeResume : result})
    bestResume = max(resumeScoreHolder, key = resumeScoreHolder.get)
    bestResumeScore = resumeScoreHolder[bestResume]
    return bestResume, bestResumeScore

# print(getBestResume(directoryInfoTech, getPDFJobDescription(sampleJob)))
print(getBestResume(directoryPR, getPDFJobDescription(sampleJob)))


# for pdfFile in os.listdir(directoryInfoTech):
#     infoTechFiles = os.path.join(directoryInfoTech, pdfFile)
#     if os.path.isfile(infoTechFiles):
#         with open (infoTechFiles, "rb") as f:
#             pdf = pdftotext.PDF(f)
#             resumeText = "\n\n".join(pdf)
#             content = [jobText, resumeText]
#             cv = CountVectorizer()
#             matrix = cv.fit_transform(content)
#             similarity_matrix = cosine_similarity(matrix)
#             print('------ resume  ------')
#             result = str(similarity_matrix[1][0]*100)
#             print('Current Resume:' + pdfFile)
#             print('Resume matches by:'+ result + '%\n')
#             scoreHolder.update({infoTechFiles : result})

# maxResume = max(scoreHolder, key = scoreHolder.get)
# print('The best resume: ' + maxResume)
# print('Its score: ' + scoreHolder[maxResume])