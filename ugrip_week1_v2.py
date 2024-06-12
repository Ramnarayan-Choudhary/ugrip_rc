# ugrip_week1.py

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers import pipeline
import torch
import requests
import zipfile
import io
import os
import json
import datetime
from vllm import LLM, SamplingParams
from openai import OpenAI, AzureOpenAI
from huggingface_hub import login

# Config info 
gpt_models = ["gpt35turbo", "gpt4"]
open_source_models = ["meta-llama/Meta-Llama-3-70B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct",'TheBloke/Llama-2-7b-Chat-AWQ']
hf_token =  "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" 

''' Note: disabled time stamps for now '''
# timestamp = datetime.datetime.now().strftime("%H_%M_%S_%m_%d")
# path_out = f"outputs_{timestamp}"

path_out = f"outputs"
path_puzzling_data = "inputs_dataset"

# Create folder if it doesn't exist
def check_path(target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        print(f"Created folder: {target_path}")

check_path(path_out)
check_path(path_puzzling_data)


def use_llm(json_tag, source_language, prompt_names, prompts, model_name, llm):
    '''
    This function uses the LLM model to generate translations/responses for the given prompts.
    
    Parameters:
    - json_tag: The 4-digit tag used for the output file name.
    - source_language: The source language for translation.
    - prompt_names: A list of prompt names.
    - prompts: A list of format strings of the populated templates.
    - model_name: The name of the model to be used.

    Returns:
    - None

    Mistral: mistralai/Mistral-7B-v0.1  [consider using more recent version of Mistral?]
    LLaMA: meta-llama/Meta-Llama-3-70B-Instruct [check if right ver?]                         
    list of supported LLMs: https://docs.vllm.ai/en/latest/models/supported_models.html#supported-models
    '''
    
    # Call LLMs or use dummy outputs
    model_type = '' # defualt directory prefix

    # Run the respective models
    if model_name in gpt_models:
        model_type = 'gpt'
        outputs = []
        client = load_model(model_name) # [?] Does this need to be called in a for loop?
        for prompt in prompts: 
            message_text = [{"role":"system","content":""}, {"role":"user", "content": prompt}]
            completion = client.chat.completions.create(
                model=model_name, # model = "deployment_name"
                messages = message_text,
                temperature=0, #TODO: change to 0.8 to be consistent? -a
                max_tokens=2048,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                seed=7777
            )
            outputs.append(completion.to_dict()['choices'][0]['message']['content'])

    elif model_name in open_source_models:
        model_type = 'open-source'
        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=2048, seed=7777)
        outputs = llm.generate(prompts, sampling_params)

    else: # If we skip over llm, use dummy outputs
        model_type = 'dummy'
        outputs = ['output01', 'output02', 'output03']

    # Deal with each output, then create:
    # "outputs_[time_stamp]\[prompt_name]\[4-digit tag]_[target_language].txt"
    # Example: "outputs_19_22_37_06_05/base_prompt_puzzling/438d_turkish.txt"
    for idx, output in enumerate(outputs):
        prompt_name = prompt_names[idx]
        if model_name in gpt_models:
            generated_text = output
        elif model_name in open_source_models:
            prompt = output.prompt
            generated_text = output.outputs[0].text
        else: 
            prompt = "dummy_prompt_content"
            generated_text = f'dummy output: {output}'

        # prepare the new output path
        os.makedirs(os.path.join(path_out, model_type, prompt_name), exist_ok=True)
        source_lang_str = source_language.rstrip().replace(" ", "_")
        out_filename = os.path.join(path_out, model_type, prompt_name, f'{json_tag}_{source_lang_str}.txt')
        
        # TODO: fix this f.write() to alignw with the .json
        with open(out_filename, "a+") as f:
            f.write(generated_text + "\n")
            print(f"SUCCESS: {out_filename} is saved.")
    print('\n')

# Populate some prompt templates, then return
# - prompt_names: A list of prompt names (such as "base_prompt_puzzling", "longer_prompt_puzzling", etc.)
# - prompts: A list of format strings of the populated templates
def create_puzzling_prompt(language, data, eng_to_lang, lang_to_eng):

    base_prompt_puzzling = f"""This is a linguistics puzzle. Below are some expressions in {language} and their English translations. 
    {data}

    Given the above expressions, please translate the following statements:
    a) from English into {language}
    {eng_to_lang}

    b) from {language} into English.

    {lang_to_eng}
  
    Please also provide your translation responses in the format of a JSON file. It should look like this: 

    "test": [
    [
    "translation sentence 1",
    "",
    "your response"
    ], 
    [
    "translation sentence 2",
    "",
    "your response"
    ], 
    ]

    where the translation sentences come from your tasks in part (a) and (b), and your translations for each of the sentences should be placed in the "your response" field. 

    """.format(language, data, eng_to_lang, lang_to_eng)

    #---------------
    longer_prompt_puzzling = f"""This is a linguistics puzzle. Below are some expressions in the {language} language and their English translations. Your task is to carefully analyze the expressions given, and use the information from them to translate some new statements. This might involve logically reasoning about how words or parts of words are structured in {language}, what the word order could be, and how different grammatical phenomena could influence the expressions. 

    All of the information you need to do this task can be obtained from the given expressions. You do not need to use any external knowledge. 

    {data}

    Given the above expressions, please translate the following statements:
    a) from English into {language}
    {eng_to_lang}

    b) from {language} into English.
    {lang_to_eng}

    Please also provide your translation responses in the format of a JSON file. It should look like this: 

    "test": [
    [
    "translation sentence 1",
    "",
    "your response"
    ], 
    [
    "translation sentence 2",
    "",
    "your response"
    ], 
    ]

    where the translation sentences come from your tasks in part (a) and (b), and your translations for each of the sentences should be placed in the "your response" field. 

    """.format(language, data, eng_to_lang, lang_to_eng)


    #-----------------------------------
    
    cot_prompt_puzzling = f"""This is a linguistics puzzle. Below are some expressions in {language} and their English translations. 
    Your task is to carefully analyze the expressions given, and use the information from them to translate some new statements. 
    All of the information you need to do this task can be obtained from the given expressions. 

    Let's think through this carefully step by step, using logical reasoning to infer the meanings of the words and get the correct answer. 

    {data}

    Given the above expressions, please translate the following statements:
     a) from English into {language}
     {eng_to_lang}

     b) from {language} into English.
     {lang_to_eng}

    Please also provide your translation responses in the format of a JSON file. It should look like this: 

    "test": [
    [
    "translation sentence 1",
    "",
    "your response"
    ], 
    [
    "translation sentence 2",
    "",
    "your response"
    ], 
    ]

    where the translation sentences come from your tasks in part (a) and (b), and your translations for each of the sentences should be placed in the "your response" field. 

    """.format(language, data, eng_to_lang, lang_to_eng)
    # [Manual Input]
    prompt_names = ['base_prompt_puzzling', 'longer_prompt_puzzling', 'cot_prompt_puzzling']
    prompts = [base_prompt_puzzling,
               longer_prompt_puzzling,
               cot_prompt_puzzling]
    
    return prompt_names, prompts


#PLS DO NOT DELETE :) -antara

def create_phonmorph_prompt(language, data, test_data = None, problem = None, family = None):

    """test_data defaults to None because morph problems do not need it
        family only required for multilingual problems"""

    #antara testing land




    #----------------

    # STRESS

    #----------------

    if problem == 'stress':

        base_prompt_stress = f"""This is a linguistics puzzle. Below are some words in the {language} language, and a sequence of numbers (ones, zeroes, and sometimes twos), corresponding to each letter in the {language} word. 
        Your task is to carefully analyze the words given, and come up with rules to explain the pattern of 0s and 1s.
        You will then apply your rules to infer the pattern of 0s and 1s (and 2s, if they exist) in some new words. 

        Here is the data:

        {data}

        And here are the new words. Please provide a sequence of 0s and 1s (and 2s, if they exist) following your inferred rules.


        {test_data}
        
        Please provide your responses in the format of a JSON file. It should look like this: 

        "test": [
        [
        "[word 1]",
        "[your predicted sequence of numbers]",
        ""
        ], 
        [
        "[word 2]",
        "[your predicted sequence of numbers]",
        ""
        ], 
        ]""".format(language, data, test_data)
        
        #-----------

        longer_prompt_stress = f"""This is a linguistics puzzle. Provided are some words in the {language} language, and a sequence of numbers (0, 1, and sometimes 2) corresponding to each letter in the {language} word. 


        Here is some information that may help to solve the puzzle. A syllable is a unit of speech that corresponds to a sound sequence, that usually has a vowel surrounded by one or more consonants. Here are some examples of syllables: "ma" is an "open" syllable, because it ends in a vowel and it is quite short. "mang" is a "closed" syllable, because it ends in consonants and it is longer. 
        In the {language} words, for any vowel, if the syllable to which that vowel belongs has stress (a property of certain types of syllables at certain locations within a word) then the number will be 1. Otherwise, if the syllable containing that vowel is unstressed, the number will be 0. 
        For example, the pattern 0 0 0 1 means the last letter has the stress, so the last syllable would have the stress. However, 0 1 0 0 means the second letter has the stress, but not necessarily that the second syllable has the stress, because syllables can be longer than just one letter. 
        If the number 2 appears, that means that the syllable has "secondary stress" -- it is stressed, but for a shorter duration. If the number 2 does not appear in the data, then it is irrelevant and you do not have to consider it.

        Your task is to carefully analyze the words given, and come up with some rules to explain why some syllables in the word are stressed (corresponding to a letter in that syllable being marked as 1), optionally secondary-stressed (corresponding to a letter in that syllable being marked as 2) and the rest are unstressed (corresponding to a letter in that syllable being marked as 0). 
        You will then apply your rules to infer the pattern of 0s and 1s, (and optionally 2s, if they exist) in some new words. 
        Think carefully and use logical reasoning. 
        All of the rules you need to solve the problem can be inferred from the given data and the explanation provided.
    
        Here is the data:

        {data}

        And here are the new words. Please provide a sequence of 0s and 1s (and 2s, if they exist) following your inferred rules.


        {test_data}
        
        Please provide your responses in the format of a JSON file. It should look like this: 

        "test": [
        [
        "[word 1]",
        "[your predicted sequence of numbers]",
        ""
        ], 
        [
        "[word 2]",
        "[your predicted sequence of numbers]",
        ""
        ], 
        ]""".format(language, data, test_data)

        prompt_names = ['base_prompt_stress', 'longer_prompt_stress']

        prompts = [base_prompt_stress,
                  longer_prompt_stress]
        
    elif problem == 'morphology':
        
        #----------------

        # MORPHOLOGY

        #----------------

        base_prompt_morph = f"""This is a linguistics puzzle. Below are some forms of words in the {language} language. 
        Your task is to carefully analyze the word forms and come up with rules to explain how the forms are derived from each other. 
        You will then apply these rules to some new words to get their alternate forms. 
        All of the information you need to do this task can be inferred from the given words. You do not need any external information. 

        Here are the word forms:

        {data}


        Wherever there is a "?" in the JSON file, using the rules you inferred, please provide your predictions for the word form that would be in the "?".
        Please output the entire JSON file, with your solutions in the "?" fields. Please do not output anything else.  

        """.format(language, data)

        #---------------------
        
        longer_prompt_morph = f"""This is a linguistics puzzle. Below are some forms of words in the {language} language. 
        Your task is to carefully analyze the word forms and come up with rules to explain how to obtain one word form from another. 
        Here is some information that may help to solve the puzzle. The forms may differ in having different kinds of affixes like prefixes, suffixes, or infixes. They may also have word-internal vowel and consonant changes. 
        You will then apply your rules to some new words to get their alternate forms. All of the information you need to do this task can be inferred from the given words. You do not need any external information. 

        Here are the word forms:
        {data}

        Wherever there is a "?" in the JSON file, using the rules you inferred, please provide your predictions for the word form that would be in the "?".
        Please output the entire JSON file, with your solutions in the "?" fields. Please do not output anything else.  
        """.format(language, data)

        prompt_names = ['base_prompt_morph', 'longer_prompt_morph']

        prompts = [base_prompt_morph,
                  longer_prompt_morph]
    
    elif problem == 'multilingual':

        #---------------

        # MULTILINGUAL

        #---------------

        base_prompt_multiling = f"""This is a linguistics puzzle. Below are some forms of words in some related languages in the {family} language family. 
        Your task is to carefully analyze the given word forms, and come up with rules to explain the changes between the words in the different languages. You will then use these rules to predict some new word forms. 
        All of the information you need to do this task can be obtained from the given word forms. You do not need to use any external knowledge. 

        Here are the word forms:

        {data}

        Wherever there is a "?" in the input JSON file, using the rules you inferred, please provide your predictions for the word form that would be in the "?". Please output the entire JSON file, with your solutions in the "?" fields. Please do not output anything else.  
        """.format(family, data)

        #-----------------

        longer_prompt_multiling = f"""This is a linguistics puzzle. Below are some forms of words in some related languages in the {family} language family. 
        Your task is to carefully analyze the given word forms, and come up with rules to explain the changes between the words in the different languages. 
        This might involve logically reasoning about different kinds of vowel and consonant changes, and thinking about what kinds of vowels/consonants are changing -- for example, whether vowels produced in the front of the mouth are changing differently from vowels produced in the back of the mouth, or whether nasal consonants are changing differently from oral consonants.
        You will then use these rules to predict some new word forms. 
        All of the information you need to do this task can be obtained from the given word forms. You do not need to use any external knowledge. 

        Here are the word forms:

        {data}

        Wherever there is a "?" in the input JSON file, using the rules you inferred, please provide your predictions for the word form that would be in the "?". Please output the entire JSON file, with your solutions in the "?" fields. Please do not output anything else.  
        """.format(family, data)
        prompt_names = ['base_prompt_multiling', 'longer_prompt_multiling']

        prompts = [base_prompt_multiling,
                  longer_prompt_multiling]
        
    elif problem == 'transliteration':
        #-----------------

        #TRANSLITERATION

        #-------------------

        base_prompt_transl = f"""This is a linguistics puzzle. Given below are some words from the {language} language in a particular orthography (writing system) and in phonetic transcription. 

        Your task is to carefully analyze the given words and their transcriptions, and come up with rules to explain how to get the transcription from the given word form, or vice versa. 

        You will then apply your rules to some new words. All of the information you need to do this task can be obtained from the given words. 

        Here are the words and their transcriptions. For all the fields marked as "?", please use your rules to predict the entry in that field and fill it in. 

        {data}""".format(language, data)

        #--------------------

        longer_prompt_transl = f"""This is a linguistics puzzle. Given below are some words from the {language} language in a particular orthography, or system of writing. For each word, there is also a phonetic transcription, which explains how the word is actually pronounced. 

        Your task is to carefully analyze the given words and their transcriptions, and come up with rules to explain how to get the transcription from the given word form, or vice versa. 
        In order to solve this puzzle, you will have to carefully think about how different sounds are realized and produced in different contexts. For example, the s in "cats" [s] and "dogs" [z] sounds different, because of the surrounding sounds. 
        You will then apply your rules to some new words. All of the information you need to do this task can be obtained from the given words. 

        Here are the words and their transcriptions. For all the fields marked as "?", please use your rules to predict the entry in that field and fill it in. 

        {data}""".format(language, data)
        prompt_names = ['base_prompt_transl', 'longer_prompt_transl']

        prompts = [base_prompt_transl,
                  longer_prompt_transl]
        
    else:
        print("Error! Need valid problem type.")


    #-------------
    #OUTPUT

    
    return prompt_names, prompts


def create_puzzling_contamination_prompt(language, data, eng_to_lang, lang_to_eng):
    translate_without_context_prompt = f"""This is a linguistics puzzle. Below are some expressions in {language} and their English translations. 
    please translate the following statements: 
    a) from {language} into English.
    {lang_to_eng}
    
    b) from English into {language}
    {eng_to_lang}

    
  
    Please also provide your translation responses in the format of a JSON file. It should look like this: 

    "test": [
    [
    "translation sentence 1",
    "",
    "your response"
    ], 
    [
    "translation sentence 2",
    "",
    "your response"
    ], 
    ]
    """.format(language, data, eng_to_lang, lang_to_eng)
    
    ask_the_language_prompt = f"""This is a linguistics puzzle. 

    Please guess what language it is?
    {data}
    
    """.format(language, data, eng_to_lang, lang_to_eng)
    
    # [Manual Input]
    prompt_names = ['translate_without_context_prompt', 'ask_the_language_prompt', "xdd"]
    prompts = [translate_without_context_prompt,
               ask_the_language_prompt]
    
    return prompt_names, prompts



# Populate the unique 4-digit tag for each problems, according to .json file names
def get_first_four_chars(filename):
    if '/' in filename:
        # Split by slash and get last element (filename)
        first_four_chars = filename.split("/")[-1][:4]
    else:
        # No slash, assume entire string is filename/extension
        first_four_chars = filename[:4]
    return first_four_chars


# Load data from the PuzzLing dataset .zip files
def init_puzzling_data():
    '''
    dataset format:
    a list of :
    'source_language'
    'target_language'
    'meta'
    'train' : a pair of source and target in a list
    'test' : a pair of source and target in a list (one is empty)
    '''

    # Path for the PuzzLing dataset to download
    url = 'https://ukplab.github.io/PuzzLing-Machines/data/public_data_dev.zip'
    directory = path_puzzling_data
    
    def download_and_extract_zip(url, directory):
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Check that the request was successful

        # Create a ZipFile object from the response content
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(directory)

    def read_puzzling_dataset(directory="None"):
        json_files = [file for file in os.listdir(directory) if file.endswith('.json')]
        json_tags = []
        json_contents = []

        for file in json_files:
            file_path = os.path.join(directory, file)
            json_tag = get_first_four_chars(file_path)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                json_tags.append(json_tag)
                json_contents.append(data)

        return json_tags, json_contents

    # Download and extract the zip file
    download_and_extract_zip(url, path_puzzling_data)

    # this has all the problems
    puzzling_problem_tags, puzzling_problem_set = read_puzzling_dataset(directory)
    return puzzling_problem_tags, puzzling_problem_set

def init_puzzling_ground_truth():
    '''
    download answer from fround truth
    '''

    # Path for the PuzzLing dataset to download
    url = 'https://ukplab.github.io/PuzzLing-Machines/data/public_reference_data_dev.zip'
    directory = "puzzling_answer"
    
    def download_and_extract_zip(url, directory):
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Check that the request was successful

        # Create a ZipFile object from the response content
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(directory)
            
    download_and_extract_zip(url, directory)

    return ""

# Load a single JSON file and return the puzzling data
def init_puzzling_data_from_json(file_path):
    # for code dependency reasons, the outputs have to be a 1-element list
    json_tags = []
    json_contents = []
     
    json_tags.append(get_first_four_chars(file_path))
    with open(file_path, 'r') as f:
        json_contents.append(json.load(f))
    return json_tags, json_contents


# Load the speficied LLM model
def load_model(model_name):
    if model_name in gpt_models:
        client = AzureOpenAI(
            azure_endpoint = "https://cullmsouthindia.openai.azure.com/", 
            api_key="037155e1b16a432fa836637370eca0e3",  
            api_version="2024-02-15-preview"
        )
    elif model_name in open_source_models:
        login(token=hf_token)   
        client = LLM(model=model_name, gpu_memory_utilization=0.7)

    return client


# Prepare the prompts, then run the LLM model
def feed_problems_to_LLM(puzzling_problem_tags, puzzling_problem_set, model_name, is_contamination_check = False):
    llm = None
    if not model_name in gpt_models:
        llm = load_model(model_name)
    # pre-process and prompt preparation
    for idx, data in enumerate(puzzling_problem_set):
        source_language = data['source_language']
        # target_language = data['target_language'] # default in english
        # meta = data['meta'] # not really needed
        source = ""
        target = ""
        source_and_target = ""

        for item in data['train']:
            source += item[0] + "\n"
            target += item[1] + "\n"
            source_and_target += item[0] + "\t" + item[1] + "\n"
        
        eng_to_lang = ""
        lang_to_eng = ""
            
        for item in data['test']:
            if item[2] == "<":
                eng_to_lang += item[1] + "\n"
            else:
                lang_to_eng += item[0] + "\n"
                
        json_tag = puzzling_problem_tags[idx]
        if is_contamination_check:
            prompt_names, prompts = create_puzzling_contamination_prompt(language=source_language, data=source_and_target, eng_to_lang=eng_to_lang, lang_to_eng=lang_to_eng)
        else:
            prompt_names, prompts = create_puzzling_prompt(language=source_language, data=source_and_target, eng_to_lang=eng_to_lang, lang_to_eng=lang_to_eng)
        
        # Print human-readable prompts
        # print("-------------- PROMPT IS HERE ----------------")
        # for prompt in prompts:
        #     print('--------------- PROMPT LINE ---------------')
        #     print(prompt)
        #     print('--------------- DIV LINE')

        # Pass the prompts into llm
        use_llm(json_tag, source_language, prompt_names, prompts, model_name, llm)
    

# [TODO] [IN-PROGRESS] Load the phonological generalizations data
def init_phonological_generalizations_data(directory=None, output_dir=None):
    '''
    This function initializes and categorizes phonological generalizations data from a given directory.
    It returns a dictionary where keys are the problem types (morphology, transliteration, stress, multilingual) 
    and values are lists of corresponding problem datasets. It also saves each problem as a separate JSON file
    with the source language and type included in the filename.
    '''
    url = 'https://github.com/saujasv/phonological-generalizations.git'
    json_dir = 'phonological-generalizations/data/problems'

    # Clone the repository if it does not exist
    if not os.path.exists(json_dir):
        os.system(f"git clone {url}")
    
    # Dictionary to hold categorized problems
    categorized_problems = {
        'morphology': [],
        'transliteration': [],
        'stress': [],
        'multilingual': []
    }
    
    
    problem_data_list = []
    # Check if the directory exists
    if os.path.exists(json_dir):
        # Load all JSON files in the directory
        for filename in os.listdir(json_dir):
            if filename.endswith('.json'):
                with open(os.path.join(json_dir, filename), 'r') as file:
                    problem = json.load(file)
                    problem_data_list.append(problem)
                    
    for problem_data in problem_data_list:
        # Categorize the problem
        if problem_data['type'] == 'morphology':
            categorized_problems['morphology'].append(problem_data)
        elif problem_data['type'] == 'transliteration':
            categorized_problems['transliteration'].append(problem_data)
        elif problem_data['type'] == 'stress':
            categorized_problems['stress'].append(problem_data)
        elif problem_data['type'] == 'multilingual':
            categorized_problems['multilingual'].append(problem_data)
        else:
            print(f"Unknown problem type: {problem_data['type']}")

    return categorized_problems

def split_data(data):
    train_data = []
    test_data = []
    for item in data['data']:
        bool_check = False
        for unit in item:
            if unit == "?":
                bool_check = True
        if bool_check:
            test_data.append(item)
        else:
            train_data.append(item)
            
    return test_data, train_data


def masked_data(data):
    for item in data['data']:
        unmasked_cnt = 0
        for unit in item:
            if unit != "?" and unit != "":
                unmasked_cnt += 1
        for idx, unit in enumerate(item):
            if unit != "?" and unit != "":
                if unmasked_cnt > 1:
                    item[idx] = "[MASK]"
                    unmasked_cnt -= 1
            
    return data

# TODO, Create more prompt for each type of problem, and maybe change the saving format
def feed_problems_to_LLM_phonology(phonology_problem_set, model_name, is_contamination_check = False):
    llm = None
    if model_name in open_source_models:
        llm = load_model(model_name)
    for idx, data in enumerate(phonology_problem_set['morphology']):
        #hello here are my comments abt formatting!!
        #for morphology: we have to ,/feed the data as a single raw json file, no train/test split
        
        language = data['languages'][0]
        data = data['data']
        prompt_names, prompts = create_phonmorph_prompt(language=language, data=data, problem='morphology')
        
        tags = f"morphology {idx}".format(idx) 
        use_llm(tags, language, prompt_names, prompts, model_name, llm)
        
    for idx, data in enumerate(phonology_problem_set['transliteration']):
        #TBD still working on prompts for this 
        language = data['languages'][0]
        data = data['data']
        prompt_names, prompts = create_phonmorph_prompt(language=language, data=data, problem='transliteration')
        
        tags = f"transliteration {idx}".format(idx) 
        use_llm(tags, language, prompt_names, prompts, model_name, llm)

    for idx, data in enumerate(phonology_problem_set['stress']):

        #for this we feed in separate train and test data, since the model has to be evaluated on the test stress patterns
        language = data['languages'][0]
        train, test = split_data(data)
        prompt_names, prompts = create_phonmorph_prompt(language=language, data=train, test_data=test, problem='stress')
        tags = f"stress {idx}".format(idx) 
        
        use_llm(tags, language, prompt_names, prompts, model_name, llm)
    for idx, data in enumerate(phonology_problem_set['multilingual']):
        #for this we don't feed in the languages -- we feed in the language FAMILY 
        # i think we can just get this from data['family'][0] or sth -- there is a parameter for this 
        # and we similarly DON'T feed in test and train data, just the data file as a raw json!
        language = data['languages'][0]
        family = data['families'][0]
        data = data['data']
        prompt_names, prompts = create_phonmorph_prompt(language=language, data=data, family=family, problem='multilingual')
        tags = f"multilingual {idx}".format(idx) 
        use_llm(tags, language, prompt_names, prompts, model_name, llm)

        #hello for transliteration
        #you pass in the data, no need split ok? ok ok
        #just liek the morph
        #ya no need to split :)
        


def main():
    # [INPUT] This could be any of the GPTs or Llama models. Ex: 'gpt-35-turbo'
    # model_name = 'meta-llama/Meta-Llama-3-70B-Instruct' TheBloke/Llama-2-7b-Chat-AWQ
    # model_name = 'gpt-35-turbo'
    
    ''' If you don't want to use any model, use "dummy" as the model name.'''
    model_name = 'gpt35turbo' 
    model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'

    # Toggle this for using .zip files or a single .json file
    bool_use_url_zip_files = True
    
    if bool_use_url_zip_files == True: # Loads all the PuzzLing dataset problems from the zip file 
        puzzling_data_tags, puzzling_problem_set = init_puzzling_data()
    else: # Use the local .json file [CAUTION] This only works on local machine rather than live code
        this_json_path = 'ed4b_tshluba_data.json'
        puzzling_data_tags, puzzling_problem_set = init_puzzling_data_from_json(this_json_path)

    feed_problems_to_LLM(puzzling_data_tags, puzzling_problem_set, model_name)


if __name__ == "__main__":
    main()
    