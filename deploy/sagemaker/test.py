import boto3
import json
import sagemaker
from deploy_env import DeployEnv
import time 
env = DeployEnv()

print("Attempting to invoke model_name=%s / env=%s..." % (env.setting('model_name'), env.current_env()))

payload ={'query':'i am samip'} 
print(env.isDeployed(),'dep')
# https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker-runtime.html#SageMakerRuntime.Client.invoke_endpoint
start=time.time()
response = env.runtime_client().invoke_endpoint(
    EndpointName=env.setting("model_name"),
    ContentType="application/json",
    Accept="application/json",
    Body=json.dumps(payload)
)



# print("Response=",response['Body'].read())
response_body = json.loads(response['Body'].read())
# res=json.dumps(response_body)
for c in response_body:
    print(c)
    print('-------------')
# print(json.dumps(response_body, indent=4))
end=time.time()
print('time elapsed: ',end-start)



# import json 
# import boto3
# import os 

# print('samip')
# endpoint=os.environ['ENDPOINT_N']
# runtime=boto3.client('sagemaker-runtime')

# def lambda_handler(event, context):
#     # # data=json.dumps(json.loads(event))
#     # payload=event
#     # print('payload',payload)
#     # response=runtime.invoke_endpoint(
#     #     EndpointName=endpoint,
#     #     ContentType="application/json",
#     #     Accept="application/json",
#     #     Body=payload
#     # )
#     # response_c=json.loads(response['Body'].read().decode("utf-8"))
#     # # print(response)
#     response_c="samip"
#     return response_c