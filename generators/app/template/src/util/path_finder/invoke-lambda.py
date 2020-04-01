import boto3
import json

#%%
payload = b"""{
    "file_name" : "test"
}"""

client = boto3.client('lambda')

response = client.invoke(
    FunctionName="returnJointMetrics-dev-nextiles",
    InvocationType="RequestResponse",
    Payload=payload
)

#%%
json_object = json.loads(response['Payload'].read())
# print(json.dumps(json_object))
print(json.dumps(json_object))
