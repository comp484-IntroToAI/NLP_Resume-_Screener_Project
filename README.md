Project Title: Resume Screening Libraries and Methods 

Authors: Isaac Wan and Jack Keller

Description:

Libraries and Dependencies:
- Sci-Kit Learn
- BERT Model and Tokenizer from Transformer (requires separate Jupyter Notebook)
- pdftotext (requires the poppler to use, see folder)
- docx2text

Recommended setup instructions:
1. Create a resume folder to hold all of your resume files and a job description folder to hold all of your job descriptions.
2. Within the resume folder and job description folder, we recommend creating .pdf and .docx subfolders to help organize your resumes and job descriptions by file type.
3. Download your resumes and job descriptions from online and place them into their correctly label folders.

How to run Sci-kit Learn .docx Resume Screener:
1. In the BasicdocxSKLEARN.py, edit the 'resume' and 'job_description' variables to the correct file path where the resume and job description are.
2. Click the 'Run' button at the top of your IDE and wait for it to return your resume's results.
 
How to run Sci-kit Learn .pdf Resume Screener:
1. In the BasicdocxSKLEARN.py, edit the 'resume' and 'job_description' variables to the correct file path where the resume and job description are.
2. Click the 'Run' button at the top of your IDE and wait for it to return your resume's results.


How to run BERT Resume Screener:
1. Create a Jupyter Notebook for the BERT.py file.
2. Create two folders: job descriptions and resumes.
3. Edit the file path in BERT.py to the correct file path where the resume and job description are.
4. Run the BERT.py file and wait for it to return your resume's results for the given job description.
