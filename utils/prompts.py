BASE_SYSTEM_PROMPT1 = """You are an assistant, tasked with generating a realistic sentence that a person with specified sociodemographics would use. 
The sentence must contain a specific word.
"""

BASE_USER_PROMPT1 = """The sociodemographics of the person are: {sociodemographics}. 
The word that must be included in the sentence is: "{word}". 

Instructions:
- The generated sentence must be in English.
- The specified word must be included in the sentence.
- Your answer should contain only the sentence, without any additional text or explanations.
- Do not repeat the sociodemographics or the task in the answer.
- Keep a natural tone that reflects how a person with those sociodemographics would speak.


Please write a realistic sentence that reflects how a person with those sociodemographics would speak.
"""

BASE_SYSTEM_PROMPT2 = """You are a person with the following sociodemographics: {sociodemographics}. 
You are asked to write a sentence that contains the word "{word}". 
The sentence should be realistic and reflect how a person with those sociodemographics would speak.
"""

BASE_USER_PROMPT2 = """Please write a realistic sentence that contains the word "{word}" that reflects how a person with your sociodemographics would speak.
"""


def prepare_input_texts(sociodemographics_list, word_list, prompt_template):
    pass
    # Call prepare_input_text for each entry

def prepare_input_text(sociodemographics, word, prompt_template):
    BASE_SYSTEM_PROMPT = BASE_SYSTEM_PROMPT1 if prompt_template == "prompt1" else BASE_SYSTEM_PROMPT2 if prompt_template == "prompt2" else None
    BASE_USER_PROMPT = BASE_USER_PROMPT1 if prompt_template == "prompt1" else BASE_USER_PROMPT2 if prompt_template == "prompt2" else None
    
    # Format the system and user prompts with the provided sociodemographics and word
    system_prompt = BASE_SYSTEM_PROMPT.format(sociodemographics=sociodemographics, word=word)
    user_prompt = BASE_USER_PROMPT.format(sociodemographics=sociodemographics, word=word)

    return system_prompt, user_prompt
