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
directoryStudentResumes = 'NLP Resume Files/Student Resumes/'
directoryCleanResumes = 'NLP Resume Files/Cleaned Resumes/'


sampleJob = "NLP Resume Files/Jobs/Sample Job Description.pdf"
sampleJob2 = "NLP Resume Files/Jobs/Job-Description2.txt"
sampleResume = "NLP Resume Files/Fullstack-Developer-Resume.txt"

list = ["the", "a", "about", "above", "actually", "after", "again", "against", "all", "almost", "also", "although", "always", "am", "an", "and", "any", "are", "as", "at",
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

def getPDFJobDescription(jobPath):
    with open (jobPath, "rb") as j:
        jobText = pdftotext.PDF(j)
        completejobDesciption = "\n\n".join(jobText)
    return completejobDesciption

def getBestResume(resumeDirectory, jobText):
    resumeScoreHolder = dict()
    for file in os.listdir(resumeDirectory):
        completeResume = os.path.join(resumeDirectory, file)
        if os.path.isfile(completeResume):
            with open (completeResume, "r") as singleResume:
                readResume = singleResume.read()
                # pdf = pdftotext.PDF(f)
                # resumeText = "\n\n".join(pdf)
                content = [jobText, readResume]
                cv = CountVectorizer(stop_words=list)
                matrix = cv.fit_transform(content)
                similarity_matrix = cosine_similarity(matrix)
                print('------ new resume  ------')
                result = str(similarity_matrix[1][0]*100)
                print('Current Resume:' + file)
                print('Resume matches by:'+ result + '%\n')
                resumeScoreHolder.update({completeResume : result})
    bestResume = max(resumeScoreHolder, key = resumeScoreHolder.get)
    bestResumeScore = resumeScoreHolder[bestResume]
    return bestResume, bestResumeScore

# print(getBestResume(directoryInfoTech, getPDFJobDescription(sampleJob)))
# print(getBestResume(directoryChef, getPDFJobDescription(sampleJob)))

print(getBestResume(directoryCleanResumes, sampleJob2))



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