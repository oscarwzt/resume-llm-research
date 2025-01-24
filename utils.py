import re
import os
import random
from multiprocessing import Pool, cpu_count
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import dot
from numpy.linalg import norm

GPT_35 = "gpt-3.5-turbo"
GPT_4 = "gpt-4"



def getResume(name = None, res_num = None, path = "resume_corpus"):
    if res_num is None:
        res_num = random.randint(0, len(os.listdir(path)) - 1)
        
    resume = name if name else os.listdir(path)[res_num]
    file_path = os.path.join(path, resume)
    
    with open(file_path, 'r', encoding='utf-8', errors="replace") as f:
        text = f.read()
    
    return f"{text}"

def clean_text(html_text):
    # Convert HTML tags and character entities to newlines
    pattern = r'<[^>]+>|&[a-z]+;'
    cleaned = re.sub(pattern, '\n', html_text)
    
    # Remove the replacement character
    cleaned = cleaned.replace("�", "")
    
    return cleaned

def format_prompt_llama2(sysIns, instruction, resume):
    return f"""<s>[INST] <<SYS>>
{sysIns}
<</SYS>>

{instruction}
{resume} [/INST]"""

def format_prompt_alpaca(instruction, resume):
	return f"""### Instruction:
{instruction}

### Input:
{resume}
"""


def convert_spans(text):
    # Replace patterns where there are consecutive tags with newline + content
    def replace_consecutive_tags(match):
        # Extract content inside the tags
        contents = re.findall(r'<span class="hl">(.*?)</span>', match.group(0))
        return '\n' + ' '.join(contents)+" "

    # Apply the replacement for consecutive tags
    cleaned = re.sub(r'(<span class="hl">.*?</span>[\s]*)+', replace_consecutive_tags, text)
    
    # Remove any remaining tags
    cleaned = re.sub(r'<span class="hl">|</span>', '', cleaned)
    
    # Remove � character
    cleaned = cleaned.replace("�", "-")
    
    return cleaned.strip()

res_format = """{
    "Basic_Info": {
        "Current_Title": "",
        "Current_Company": "",
        "Location": "",
        "Bio": ""
    },
    "Experience": [
        {
            "Job_Title": "",
            "Company": "",
            "Location": "",
            "Start_Date": "",
            "End_Date": "",
            "Responsibilities": ""
        }
    ],
    "Education": [
        {
            "Degree": "",
            "Field": "",
            "Institution": "",
            "Location": "",
            "Graduation_Date": ""
        }
    ],
    "Projects": [
        {
            "Project_Title": "",
            "Description": "",
            "Date": ""
        }
    ],
    "Skills": [],
    "Technical_Skills": [],
    "Links": [],
    "Certifications": [
        {
            "Certification_Title": "",
            "Issuing_Organization": "",
            "Date_Issued": ""
        }
    ],
    "Awards": [
        {
            "Award_Title": "",
            "Issuing_Organization": ""
        }
    ],
    "Publications": [
        {
            "Publication_Title": "",
            "Date": ""
        }
    ]
}
"""

INSTRUCTION = f"Please parse the resume into the following JSON format: \n{res_format}"
SYSINS = "You are a detail-oriented resume parser who does not miss any information in resumes."

def getLlamaPrompt(resume, sysIns = SYSINS, instruction = INSTRUCTION):
    
    return format_prompt_llama2(sysIns, instruction, convert_spans(getResume(resume)).replace('"', ""))

count_tokens = lambda x: len(tokenizer.encode(x))

def getCleanResume(resume):
    return convert_spans(getResume(resume)).replace('"', "").replace("- ", " ").replace("  ", " ").replace("?", "")


def worker(file_path):
    file_path = file_path.replace("resume_corpus/", "", 1)
    content = getCleanResume(file_path)
    return content, file_path

def create_dict_from_files(directory):
    file_paths = [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files]

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(worker, file_paths)
    
    return {content: os.path.basename(file_path) for content, file_path in results}


def compute_jaccard(text1, text2):
    # Tokenize the texts and convert to sets
    set1 = set(text1.split())
    set2 = set(text2.split())
    
    if not set1 and not set2:  # if both sets are empty
        return 1.0
    
    return len(set1.intersection(set2)) / len(set1.union(set2))


def cosine_similarity(vecA, vecB):
    if norm(vecA) == 0 or norm(vecB) == 0:
        return 0
    return dot(vecA, vecB) / (norm(vecA) * norm(vecB))

def compute_cosine_similarity(text1, text2):
    if len(text1) == 0 or len(text2) == 0:
        return 2
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    return cosine_similarity(vectorizer[0].toarray()[0], vectorizer[1].toarray()[0])

def load_conversation_into_messages(conversation_path):
    text = open(conversation_path, "r").read()
    text = text.replace("Chatbot:", "ASSISTANT:")
    text = text.replace("You:", "USER:")
    lines = text.split('\n')
    # Creating the list in the desired format
    messages = []
    current_content = ""
    current_role = ""

    for line in lines:
        if line.startswith("USER:"):
            if current_role:  # If there is a previous role, append the content to messages
                messages.append({"role": current_role, "content": current_content.strip()})
            current_role = "user"
            current_content = line[len("USER:"):].strip()  # Strip the "You:" part and leading/trailing spaces
        elif line.startswith("ASSISTANT:"):
            if current_role:  # If there is a previous role, append the content to messages
                messages.append({"role": current_role, "content": current_content.strip()})
            current_role = "assistant"
            current_content = line[len("ASSISTANT:"):].strip()  # Strip the "Chatbot:" part and leading/trailing spaces
        else:
            # If the line does not start with "You:" or "Chatbot:", append it to the current content
            current_content += "\n" + line

    # Add the last content to the messages list
    if current_role and current_content:
        messages.append({"role": current_role, "content": current_content.strip()})
    return messages


def create_resume_from_json(resume_json, read = True):
    """Re-construct the textual resume from the provided JSON data."""
    
    def flatten_and_convert(data):
        """Flatten and convert data to string."""
        if isinstance(data, list):
            return ' '.join(map(str, data))
        return str(data)

    resume_text = []

    # Basic Info
    basic_info = resume_json.get('Basic_Info', {})
    title = flatten_and_convert(basic_info.get('Current_Title', ''))
    company = flatten_and_convert(basic_info.get('Current_Company', ''))
    resume_text.append(title + " at " + company)
    resume_text.append(flatten_and_convert(basic_info.get('Location', '')))
    bio = flatten_and_convert(basic_info.get('Bio', ''))
    if bio:
        resume_text.append(bio)
    resume_text.append("\nExperience:")

    # Experience
    for exp in resume_json.get('Experience', []):
        job_title = flatten_and_convert(exp.get('Job_Title', ''))
        company = flatten_and_convert(exp.get('Company', ''))
        start_date = flatten_and_convert(exp.get('Start_Date', ''))
        end_date = flatten_and_convert(exp.get('End_Date', ''))
        responsibilities = flatten_and_convert(exp.get('Responsibilities', ''))
        resume_text.append(job_title + " at " + company + " " + start_date + " to " + end_date)
        resume_text.append(responsibilities)

    # Education
    resume_text.append("\nEducation:")
    for edu in resume_json.get('Education', []):
        degree = flatten_and_convert(edu.get('Degree', ''))
        field = flatten_and_convert(edu.get('Field', ''))
        institution = flatten_and_convert(edu.get('Institution', ''))
        resume_text.append(degree + " in " + field + " from " + institution)

    # Skills
    skills = flatten_and_convert(resume_json.get('Skills', []))
    if skills:
        resume_text.append("\nSkills:")
        resume_text.append(skills)

    # Links
    links = flatten_and_convert(resume_json.get('Links', []))
    if links:
        resume_text.append("\nLinks:")
        resume_text.append(links)

    # Additional Information
    additional_info = flatten_and_convert(resume_json.get('Additional_Information', ''))
    if additional_info:
        resume_text.append("\nAdditional Information:")
        resume_text.append(additional_info)

    # Joining all the parts to form a single string
    resume_formatted = "\n".join(resume_text)
    
    if not read:
        resume_formatted = resume_formatted.replace("\n", " ")
    
    return resume_formatted