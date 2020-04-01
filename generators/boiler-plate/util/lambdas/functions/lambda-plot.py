import json
import os
import boto3
import io

# from layers
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def lambda_handler(event, context):
    
    # initialize S3 variables
    # BUCKET = event['Records'][0]['s3']['bucket']['name']
    # BUCKET = os.environ['BUCKET']
    BUCKET = "nextiles-playground"
    DATA_PREFIX = "raw/"
    META_PREFIX = "meta/"
    
    # S3 access
    client = boto3.client('s3')
    file_name = event['file_name']
    # file_name = event['Records'][0]['s3']['object']['key']
    file_data_type = ".csv"
    file_meta_type = ".json"
    
    # get relevant data files
    try:
        data_resp = client.get_object(Bucket=BUCKET,
                                      Key=DATA_PREFIX+file_name+file_data_type
                                      )
    except:
        return return_response(404, "no data .csv object found")
    
    try:
        meta_resp = client.get_object(Bucket=BUCKET,
                                      Key=META_PREFIX+file_name+file_data_type
                                      )
    except:
        return return_response(400, "no data .csv object found")
    
    # read in data
    data_frame = pd.read_csv(data_resp['Body'])
    try:
        pressure_data = data_frame['6'].dropna()
    except:
        return return_response(400, "no pressure data found")
    
    # set up plotting
    fig, ax = plt.subplots(1,1)
    ax.plot(pressure_data)
    
    # put saved data
    SAVE_FIG_PREFIX = "raw-plots/"
    
    # save figure
    data_fig = save_image(fig)
    s3_save_location = SAVE_FIG_PREFIX+"{}-data.png".format(file_name
    
    # put assets into S3 bucket
    client.put_object(
        Bucket=BUCKET,
        Body=data_fig,
        ContentType="image/png",
        Key=)
    )
    
    return return_response(200, 
                           data_dump_json)

#%%
def save_image(fig):
    imgio = io.BytesIO()
    fig.savefig(imgio, format="png")
    imgio.seek(0)    
    return imgio

def return_response(statusCode, body):
    return {
        "statusCode": 200,
        "body" : body
        }

#%%
# TEST
if __name__ == "__main__":
    # file_name = "NTEST-KNE_1_1_strength-training_02-08-2020_1581176957359"
    # file_name = "NTEST-KNE_1_1_NONE_02-08-2020_1581171193161"
    # file_name = "NTEST-KNE_1_1_strength-training_02-08-2020_1581176856434"
    # file_name = "NTEST-KNE_1_1_strength-training_02-08-2020_1581174777001"
    file_name = "test"
    event = {
        "file_name" : file_name
    }
    context = []
    lambda_handler(event, context)