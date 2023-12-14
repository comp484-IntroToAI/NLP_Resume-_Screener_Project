import docx2txt
directoryInfoTech = '/Users/jackkeller/Desktop/484F23/project-jack_isaac/NLP Resume Files/data/INFORMATION-TECHNOLOGY'
import os
import itertools
import math
import re
import pdftotext
import torch as torch
import torch.nn.functional as Y

resume = docx2txt.process('/Users/jackkeller/Desktop/484F23/project-jack_isaac/NLP Resume Files/Cleaned Resumes docx/JackKeller_Resume.docx')
# resume = docx2txt.process('content/Keller_Tech_Resume9.18.23.docx')
job_description = "**Job Title: Entry-Level Software Engineer* [Company Name] is a dynamic and innovative software development company committed to creating cutting-edge solutions that redefine industries. We pride ourselves on fostering a collaborative and inclusive work environment where creativity and technical excellence thrive. As we continue to expand our team, we are seeking passionate and talented individuals to join us on our journey. **Position Overview:** We are looking for an entry-level software engineer to contribute to the design, development, and maintenance of our software applications. As a key member of our engineering team, you will work closely with experienced developers, participating in various stages of the software development lifecycle. This is an excellent opportunity for a recent graduate or an individual with limited professional experience to grow their skills in a supportive and challenging environment.**Responsibilities:** Collaborate with cross-functional teams to understand project requirements and specifications. Assist in the design and implementation of software solutions, following best practices and coding standards. Write clean, efficient, and well-documented code. Participate in code reviews to ensure the quality of the codebase. Debug and troubleshoot software defects and issues. Learn and adapt to new technologies and programming languages as needed. Contribute ideas to improve existing processes and workflows. **Qualifications:** Bachelor’s degree in Computer Science, Software Engineering, or a related field. Strong understanding of programming fundamentals and object-oriented design. Proficiency in at least one programming language (e.g., Java, C++, Python, etc.). Knowledge of software development tools and methodologies. Excellent problem-solving and analytical skills. Good communication and teamwork abilities. Demonstrated passion for software development through personal projects, internships, or coursework.**Preferred Skills:**  Familiarity with web development technologies (HTML, CSS, JavaScript). Experience with version control systems (Git, SVN). Understanding of database concepts and SQL. 4. Knowledge of software testing principles. 5. Exposure to agile development methodologies. **Perks and Benefits:** 1. Competitive salary and performance-based bonuses. 2. Comprehensive health, dental, and vision insurance. 3. Flexible work hours and remote work options. 4. Opportunities for professional development and career growth. 5. Fun and collaborative work environment with team-building activities."
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
                resume = pdftotext.PDF(f)
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

job2Text = open("/Users/jackkeller/Desktop/484F23/project-jack_isaac/NLP Resume Files/Jobs/Job-Description.txt")

sampleJob2 = job2Text.read()
# print(sampleJob2)
print(getBestResume(directoryInfoTech, sampleJob2))