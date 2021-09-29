from os import truncate
import boto3
import json
import sagemaker
from deploy_env import DeployEnv
import time 
env = DeployEnv()

print("Attempting to invoke model_name=%s / env=%s..." % (env.setting('model_name'), env.current_env()))
print(env.isDeployed(),'dep')
while True:
    query=input('Enter your query: ')
    payload ={'query':query} 
    start=time.time()
    response = env.runtime_client().invoke_endpoint(
        EndpointName=env.setting("model_name"),
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload)
    )
    response_body = json.loads(response['Body'].read())
    for c in response_body:
        print(c)
        print('-------------')
    end=time.time()
    print('time elapsed: ',end-start)
