from langchain.llms import OpenAI
# decouple to read .env variables(OpenAI Key)
from decouple import config
# import openAI from langChain
from langchain.llms import OpenAI
# import prompt template
from langchain.prompts import PromptTemplate 


# create the prompt
prompt_template: str = """/
You are a vehicle mechanic, give responses to the following/ 
question: {question}. Do not use technical words, give easy/
to understand responses.
"""

prompt = PromptTemplate.from_template(template=prompt_template)

# format the prompt to add variable values
prompt_formatted_str: str = prompt.format(
    question="Why won't a vehicle start on ignition?")

# instantiate the OpenAI intance
llm = OpenAI(openai_api_key=config("OPENAI_API_KEY"))

# make a prediction
prediction = llm.invoke(prompt_formatted_str)

# print the prediction
print(prediction)