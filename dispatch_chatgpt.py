import openai
import asyncio
from typing import Any
from utils import *
import time
from tqdm import tqdm
import json
import os
from openai.error import APIError
from json import JSONDecodeError

sysIns = """You are a detail-oriented resume parsing machine. You do not miss any information on the resume. You do not add any information that is not on the resume. You parse the resumes the users give you into this exact format: 
{
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

The responsibility section should not be a summary of what's written in the corresponding section in the original resume. It should be the exact copy, not including special characters.
"""
with open("chatGPT_API_KEY.txt") as f:
    API_KEY = f.read()
    
openai.api_key = API_KEY

async def dispatch_openai_requests(
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float ,
    max_tokens: int = None,
    presence_penalty: float = 0,
) -> list[str]:
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


# with open("resumes_groupby_tokens/resumes_400_600.txt") as f:
#     resumes = f.read().split("\n")
    
# with open("resumes_groupby_tokens/resumes_600_800.txt") as f:
#     resumes += f.read().split("\n")
    
# with open("resumes_groupby_tokens/resumes_800_1000.txt") as f:
#     resumes += f.read().split("\n")
    
# with open("resumes_groupby_tokens/resumes_1000_1200.txt") as f:
#     resumes += f.read().split("\n")
    
# with open("resumes_groupby_tokens/resumes_1200_1400.txt") as f:
#     resumes += f.read().split("\n")
    

# # check if parsed_resume_jsons/new_chatgt.json exists
# # if it does, load it
# # if it doesn't, create empty dict

# with open("parsed_resume_jsons/new_chatgpt.json", "r") as f:
#     parsed_already = json.load(f)
 
# with open("parsed_resume_jsons/new_chatgpt_failed.json") as f:
#     failed_already = json.load(f)       

# resumes = [res for res in resumes if res not in parsed_already and res not in failed_already]

# parsed_current = parsed_already
# parse_failed = failed_already

# batch_size = 20

# MAX_RETRIES = 3
# RETRY_DELAY = 10  # Delay for 10 seconds

# for ii in tqdm(range(0, len(resumes), batch_size)):
#     cur_len_failed = len(parse_failed)
#     message_list = [[{'role': "system", "content": SYSINS},
#                     {"role": "user", "content": INSTRUCTION + getCleanResume(res)}]
#                 for res in resumes[ii:ii+batch_size]]

#     retries = 0
#     success = False
    
#     has_failed_parse = False
    
#     while retries < MAX_RETRIES and not success:
#         try:
#             predictions = asyncio.run(
#                 dispatch_openai_requests(
#                     messages_list=message_list,
#                     model="gpt-3.5-turbo",
#                     temperature=0,
#                     presence_penalty=-0.75
#                 )
#             )
            
#             for i, x in enumerate(predictions):
#                 extracted_resume = x['choices'][0]['message']['content']
#                 try:
#                     parsed_current[resumes[ii+i]] = eval(extracted_resume)
#                 except:
#                     has_failed_parse = True
#                     parse_failed[resumes[ii+i]] = extracted_resume
#                     print(f"Failed to parse {resumes[ii+i]}")
            
#             with open("parsed_resume_jsons/new_chatgpt.json", "w") as f:
#                 json.dump(parsed_current, f, indent=4)
                
#             if has_failed_parse:
#                 with open("parsed_resume_jsons/new_chatgpt_failed.json", "w") as f:
#                     json.dump(parse_failed, f, indent=4)

#             success = True
        
        
#         except APIError as e:
#             if e.http_status == 502:
#                 print(f"Got a 502 Bad Gateway error. Retrying in {RETRY_DELAY} seconds...")
#                 time.sleep(RETRY_DELAY)
#             else:
#                 print(f"An error occurred: {e}")
#                 break  # Exit the while loop and proceed to the next iteration
        
#         except Exception as e:
#             print(f"An unexpected error occurred: {e}")
#             break  # Exit the while loop and proceed to the next iteration
        
#         retries += 1

#     if not success:
#         print(f"Max retries reached for batch starting at index {ii}. Moving to the next batch.")

batch_size = 40
parse_failed = dict()
parsed_current = dict()
resumes = ['07416.txt',
 '00820.txt',
 '13015.txt',
 '00116.txt',
 '26453.txt',
 '21699.txt',
 '00014.txt',
 '03070.txt',
 '10854.txt',
 '29317.txt',
 '01971.txt',
 '29204.txt',
 '04981.txt',
 '11438.txt',
 '24222.txt',
 '20088.txt',
 '15854.txt',
 '07455.txt',
 '16515.txt',
 '27182.txt',
 '11724.txt',
 '18789.txt',
 '22485.txt',
 '22041.txt',
 '08872.txt',
 '26438.txt',
 '08293.txt',
 '25730.txt',
 '27537.txt',
 '21385.txt',
 '11685.txt',
 '03771.txt',
 '01396.txt',
 '03114.txt',
 '11800.txt',
 '02466.txt',
 '17266.txt',
 '20773.txt',
 '17230.txt',
 '24578.txt']
for ii in tqdm(range(0, len(resumes), batch_size)):
    cur_len_failed = len(parse_failed)
    message_list = [[{'role': "system", "content": sysIns},
                    {"role": "user", "content": "Please parse the following resume:\n"  + getCleanResume(res)}]
                for res in resumes[ii:ii+batch_size]]
    predictions = asyncio.run(
                dispatch_openai_requests(
                    messages_list=message_list,
                    model="gpt-4",
                    temperature=0,
                    presence_penalty=-0.75
                )
            )
            
    for i, x in enumerate(predictions):
        extracted_resume = x['choices'][0]['message']['content']
        try:
            parsed_current[resumes[ii+i]] = eval(extracted_resume)
        except:
            has_failed_parse = True
            parse_failed[resumes[ii+i]] = extracted_resume
            print(f"Failed to parse {resumes[ii+i]}")
    
    with open("gpt4_parsing.json", "w") as f:
        json.dump(parsed_current, f, indent=4)