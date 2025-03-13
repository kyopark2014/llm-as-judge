
import info
import utils
import boto3
from botocore.config import Config
from langchain_aws import ChatBedrock

logger = utils.CreateLogger("chat")

model_name = "Claude 3.7 Sonnet"
model_type = "claude"
models = info.get_model_info(model_name)
number_of_models = len(models)
selected_chat = 0
profile = models[selected_chat]
bedrock_region =  profile['bedrock_region']
modelId = profile['model_id']
model_type = profile['model_type']

if model_type == 'claude':
    maxOutputTokens = 4096 # 4k
else:
    maxOutputTokens = 5120 # 5k    
logger.info(f"LLM: {selected_chat}, bedrock_region: {bedrock_region}, modelId: {modelId}, model_type: {model_type}")

if profile['model_type'] == 'nova':
    STOP_SEQUENCE = '"\n\n<thinking>", "\n<thinking>", " <thinking>"'
elif profile['model_type'] == 'claude':
    STOP_SEQUENCE = "\n\nHuman:" 

parameters = {
    "max_tokens":maxOutputTokens,     
    "temperature":0.1,
    "top_k":250,
    "top_p":0.9,
    "stop_sequences": [STOP_SEQUENCE]
}

# bedrock   
boto3_bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=bedrock_region,
    config=Config(
        retries = {
            'max_attempts': 30
        }
    )
)

def get_chat():
    llm = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
        region_name=bedrock_region
    ) 

    return llm

