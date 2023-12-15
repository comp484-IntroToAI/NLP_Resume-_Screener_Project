#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pdftotext
import os
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

directoryInfoTech = 'NLP Resume Files/data/INFORMATION-TECHNOLOGY/'
directoryChef = 'NLP Resume Files/data/CHEF/'
directoryBanking = 'NLP Resume Files/data/BANKING/'
sampleResume = open('/Users/jackkeller/Desktop/484F23/project-jack_isaac/NLP Resume Files/Jobs/StudentResumeTxts/JessicaMcAllister.txt')
sampleResume = sampleResume.read()

job_description = "**Job Title: Entry-Level Software Engineer* [Company Name] is a dynamic and innovative software development company committed to creating cutting-edge solutions that redefine industries. We pride ourselves on fostering a collaborative and inclusive work environment where creativity and technical excellence thrive. As we continue to expand our team, we are seeking passionate and talented individuals to join us on our journey. **Position Overview:** We are looking for an entry-level software engineer to contribute to the design, development, and maintenance of our software applications. As a key member of our engineering team, you will work closely with experienced developers, participating in various stages of the software development lifecycle. This is an excellent opportunity for a recent graduate or an individual with limited professional experience to grow their skills in a supportive and challenging environment.**Responsibilities:** Collaborate with cross-functional teams to understand project requirements and specifications. Assist in the design and implementation of software solutions, following best practices and coding standards. Write clean, efficient, and well-documented code. Participate in code reviews to ensure the quality of the codebase. Debug and troubleshoot software defects and issues. Learn and adapt to new technologies and programming languages as needed. Contribute ideas to improve existing processes and workflows. **Qualifications:** Bachelor’s degree in Computer Science, Software Engineering, or a related field. Strong understanding of programming fundamentals and object-oriented design. Proficiency in at least one programming language (e.g., Java, C++, Python, etc.). Knowledge of software development tools and methodologies. Excellent problem-solving and analytical skills. Good communication and teamwork abilities. Demonstrated passion for software development through personal projects, internships, or coursework.**Preferred Skills:**  Familiarity with web development technologies (HTML, CSS, JavaScript). Experience with version control systems (Git, SVN). Understanding of database concepts and SQL. 4. Knowledge of software testing principles. 5. Exposure to agile development methodologies. **Perks and Benefits:** 1. Competitive salary and performance-based bonuses. 2. Comprehensive health, dental, and vision insurance. 3. Flexible work hours and remote work options. 4. Opportunities for professional development and career growth. 5. Fun and collaborative work environment with team-building activities."
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
encoder = BertModel.from_pretrained("bert-base-uncased")

def getBestResume(resumeDirectory, jobText, tokenizer, encoder):
    counter = 0
    resumeScoreHolder = dict()
    for file in os.listdir(resumeDirectory):
        completeResume = os.path.join(resumeDirectory, file)
        if os.path.isfile(completeResume):
            with open (completeResume, "rb") as f:
                pdf = pdftotext.PDF(f)
                resumeText = "\n\n".join(pdf)
                resumeText = cleanText(resumeText)
                jobText = cleanText(jobText)
                resume_embedding = BERT_encode(resumeText, tokenizer, encoder)
                job_ad_embedding = BERT_encode(jobText, tokenizer, encoder)
                val = F.cosine_similarity(resume_embedding, job_ad_embedding)
                result = str(val.item())
                print('--------New Resume--------')
                print(result)
                if (float(result) > 0.954):
                   counter = counter + 1    
                resumeScoreHolder.update({completeResume : result})
    bestResume = max(resumeScoreHolder, key = resumeScoreHolder.get)
    bestResumeScore = resumeScoreHolder[bestResume]
    print("HOLY CRAP THIS DIRECTORY HAS THIS MNAY HAT PASSED: " + str(counter))
    return bestResume, bestResumeScore
                

# Extract embeddings. BertModel has no classification head on top, so the output is just the hidden states of the last layer.
# BERT was trained such that there is a special token (called CLS or the classification token) whose representation is used for sentence classification tasks.
#.pooler_output retrieves that representation.

# In[3]:
def BERT_encode(text, tokenizer, encoder):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    return encoder(**inputs).pooler_output

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

print("INITIATE!!---------")
print(getBestResume(directoryInfoTech, job_description, tokenizer, encoder))
print("---------")
print(getBestResume(directoryBanking, job_description, tokenizer, encoder))
print("---------")
print(getBestResume(directoryChef, job_description, tokenizer, encoder))

