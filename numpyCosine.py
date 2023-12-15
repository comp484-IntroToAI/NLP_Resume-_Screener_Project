# Python program to generate word vectors using Word2Vec
# !pip install pdftotext
# import pdftotext

# importing all necessary modules
import warnings
import numpy as np
import pdftotext
import os

warnings.filterwarnings(action = 'ignore')


# Isaac's Drive Pathway
# sampleResume = open("/content/drive/MyDrive/Intro to A.I./A.I. NLP Project/NLP A.I. Resume Files/Test Resumes TXT/AlbertLuna_Resume.txt")
directoryInfoTech = 'NLP Resume Files/data/INFORMATION-TECHNOLOGY/'
directoryChef = 'NLP Resume Files/data/CHEF/'
directoryBanking = 'NLP Resume Files/data/BANKING/'

job_description = "**Job Title: Entry-Level Software Engineer* [Company Name] is a dynamic and innovative software development company committed to creating cutting-edge solutions that redefine industries. We pride ourselves on fostering a collaborative and inclusive work environment where creativity and technical excellence thrive. As we continue to expand our team, we are seeking passionate and talented individuals to join us on our journey. **Position Overview:** We are looking for an entry-level software engineer to contribute to the design, development, and maintenance of our software applications. As a key member of our engineering team, you will work closely with experienced developers, participating in various stages of the software development lifecycle. This is an excellent opportunity for a recent graduate or an individual with limited professional experience to grow their skills in a supportive and challenging environment.**Responsibilities:** Collaborate with cross-functional teams to understand project requirements and specifications. Assist in the design and implementation of software solutions, following best practices and coding standards. Write clean, efficient, and well-documented code. Participate in code reviews to ensure the quality of the codebase. Debug and troubleshoot software defects and issues. Learn and adapt to new technologies and programming languages as needed. Contribute ideas to improve existing processes and workflows. **Qualifications:** Bachelor’s degree in Computer Science, Software Engineering, or a related field. Strong understanding of programming fundamentals and object-oriented design. Proficiency in at least one programming language (e.g., Java, C++, Python, etc.). Knowledge of software development tools and methodologies. Excellent problem-solving and analytical skills. Good communication and teamwork abilities. Demonstrated passion for software development through personal projects, internships, or coursework.**Preferred Skills:**  Familiarity with web development technologies (HTML, CSS, JavaScript). Experience with version control systems (Git, SVN). Understanding of database concepts and SQL. 4. Knowledge of software testing principles. 5. Exposure to agile development methodologies. **Perks and Benefits:** 1. Competitive salary and performance-based bonuses. 2. Comprehensive health, dental, and vision insurance. 3. Flexible work hours and remote work options. 4. Opportunities for professional development and career growth. 5. Fun and collaborative work environment with team-building activities."

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

def getBestResume(resumeDirectory, jobText):
    counter = 0
    resumeScoreHolder = dict()
    for file in os.listdir(resumeDirectory):
        completeResume = os.path.join(resumeDirectory, file)
        print(file)
        if os.path.isfile(completeResume):
            with open (completeResume, "rb") as f:
                pdf = pdftotext.PDF(f)
                resumeText = "\n\n".join(pdf)
                resumeText = cleanText(resumeText)
                jobText = cleanText(jobText)
                vocab = set(resumeText.split())
                vocab2 = set(jobText.split())
                vocab = vocab.union(set(vocab2))
                vocab = list(vocab)
                vA = np.zeros(len(vocab), dtype=float)
                vB = np.zeros(len(vocab), dtype=float)
                print('------ new resume  ------')
                for w in resumeText.split():
                  try:
                    i = vocab.index(w)
                    vA[i] += 1
                  except ValueError:
                    pass
                for w in jobText.split():
                  try:
                    i = vocab.index(w)
                    vB[i] += 1
                  except ValueError:
                    pass
                cos = np.dot(vA, vB) / (np.sqrt(np.dot(vA,vA)) * np.sqrt(np.dot(vB,vB)))
                print(cos)
                if (float(cos) > 0.68) :
                    counter = counter + 1
                resumeScoreHolder.update({completeResume : cos})
    bestResume = max(resumeScoreHolder, key = resumeScoreHolder.get)
    bestResumeScore = resumeScoreHolder[bestResume]
    print("HOLY CRAP THIS DIRECTORY HAS THIS MNAY HAT PASSED: " + str(counter))
    return bestResume, bestResumeScore

print("INITIATING PROGRAM ______--------_______")
print(getBestResume(directoryInfoTech, job_description))
print("--------------------")
print(getBestResume(directoryBanking, job_description))
print("--------------------")
print(getBestResume(directoryChef, job_description))



