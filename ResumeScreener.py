pip install docx2txt
import docx2txt

job_description = docx2txt.process('/content/Sample Job Description.docx')
resume = docx2txt.process('/content/Keller_Tech_Resume9.18.23.docx')

print(resume)

content = [job_description, resume]

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
matrix = cv.fit_transform(content)

from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(matrix)

print(similarity_matrix)

print('Resume matches by: '+ str(similarity_matrix[1][0]*100)+ '%')