import os
from dotenv import load_dotenv
load_dotenv()
import boto3

aws_access_key_id = os.getenv('AWS_SES')
aws_secret_access_key = os.getenv('AWS_SECRET')
region_name = 'ca-central-1'
from_email = 'marinusdebeer@gmail.com'
to_email = 'marinusdebeer@gmail.com'

ses_client = boto3.client(
    'ses',
    region_name=region_name,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)
def send_email(subject, body, logging=True):
    # Send the email
    try:
        response = ses_client.send_email(
            Source=from_email,
            Destination={
                'ToAddresses': [to_email]
            },
            Message={
                'Subject': {
                    'Data': subject
                },
                'Body': {
                    'Text': {
                        'Data': body
                    }
                }
            }
        )
        if logging:
            print('Email sent! Message ID:', response['MessageId'])
    except Exception as e:
        print('Something went wrong:', e)
