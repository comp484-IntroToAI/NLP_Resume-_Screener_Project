import docx2txt
directoryInfoTech = '/Users/jackkeller/Desktop/484F23/project-jack_isaac/NLP Resume Files/data/INFORMATION-TECHNOLOGY'
import os
import itertools
import math
import re
import pdftotext


directoryInfoTech = 'NLP Resume Files/data/INFORMATION-TECHNOLOGY/'
# resume = docx2txt.process('content/Keller_Tech_Resume9.18.23.docx')
job_description = "NLP Resume Files\Jobs\Job-Description.txt"
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

# content = [job_description, resume]




from sklearn.feature_extraction.text import CountVectorizer


# cv = CountVectorizer(stop_words=list)
# matrix = cv.fit_transform(content)

from sklearn.metrics.pairwise import cosine_similarity


def getPDFJobDescription(jobPath):
    with open (jobPath, "rb") as j:
        jobText = pdftotext.PDF(j)
        completejobDesciption = "\n\n".join(jobText)
    return completejobDesciption

def getBestResume(resumeDirectory, jobText):
    counter = 0
    resumeScoreHolder = dict()
    for file in os.listdir(resumeDirectory):
        print(file)
        completeResume = os.path.join(resumeDirectory, file)
        if os.path.isfile(completeResume):
            with open (completeResume, "r") as f:
                resume = docx2txt.process(f)
                resumeText = "\n\n".join(resume)
                content = [jobText, resumeText]
                cv = CountVectorizer(stop_words=list)
                matrix = cv.fit_transform(content)
                similarity_matrix = cosine_similarity(matrix)
                print('------ new resume  ------')
                result = str(similarity_matrix[1][0]*100)
                if (result > 20):
                    counter = counter + 1
    bestResume = max(resumeScoreHolder, key = resumeScoreHolder.get)
    bestResumeScore = resumeScoreHolder[bestResume]
    return bestResume, bestResumeScore, counter


# print(sampleJob2)
print(getBestResume(directoryInfoTech, job_description))