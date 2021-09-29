import sys
sys.path.append(".")
import sagemaker
import os
import boto3
import re
from deploy_env import DeployEnv
from sagemaker.pytorch import PyTorchModel
from code_files import config

env = DeployEnv()

def s3_bucket_from_url(s3_url):
    return re.search('//(.+)/',s3_url).groups()[0]

def upload_model_data():
    if env.isLocal():
        return
    bucket_name = s3_bucket_from_url(env.setting('model_data_path'))
    config.logger.info("Uploading model.tar.gz to S3 bucket=%s..." % (bucket_name))
    s3 = boto3.resource('s3')
    s3.create_bucket(Bucket=bucket_name)
    return s3.Bucket(bucket_name).upload_file("build/model.tar.gz", "model.tar.gz")
    config.logger.info("\t...DONE.")

def build_model_data_file():
    return os.system("tar -czf build/model.tar.gz code_files code data model.pth logging.json")

def update_endpoint_if_exists():
    return (env.isProduction() & env.isDeployed())

def delete_endpoint_and_config():
    """
    Need to manually delete the endpoint and config because of
    https://github.com/aws/sagemaker-python-sdk/issues/101#issuecomment-607376320.
    """
    env.client().delete_endpoint(EndpointName=env.setting('model_name'))
    env.client().delete_endpoint_config(EndpointConfigName=env.setting('model_name'))
def get_execution_role(sagemaker_session=None):
    """
    Returns the role ARN whose credentials are used to call the API.
    In AWS notebook instance, this will return the ARN attributed to the
    notebook. Otherwise, it will return the ARN stored in settings
    at the project level.
    :param: sagemaker_session(Session): Current sagemaker session
    :rtype: string: the role ARN
    """
    try:
        role = sagemaker.get_execution_role(sagemaker_session=sagemaker_session)
    except ValueError as e:
       #copied from sagemaker notebook instance. change this if needed.
        role='arn:aws:iam::513354041882:role/service-role/AmazonSageMaker-ExecutionRole-20210804T180621'
    
    return role
def deploy():
    config.logger.info("Deploying model_name=%s to env=%s" % (env.setting('model_name'), env.current_env()))
    # build_model_data_file()
    # upload_model_data()
    aws_role=get_execution_role()
    pytorch_model = PyTorchModel(
        model_data = env.setting('model_data_path'),
        name = env.setting('model_name'),
        framework_version = '1.4.0',
         py_version="py3",
        role = aws_role,
        env = {"DEPLOY_ENV": env.current_env()},
        entry_point = 'deploy/sagemaker/serven.py')

    if env.isDeployed():
        delete_endpoint_and_config()
    config.logger.info("Instance type: %s" %(env.setting('instance_type'))) 
    config.logger.info("Instance no: %s" %(str(2)+"instances")) 
    predictor = pytorch_model.deploy(
        instance_type = env.setting('instance_type'),
        initial_instance_count = 2)

if __name__ == '__main__':
    deploy()
