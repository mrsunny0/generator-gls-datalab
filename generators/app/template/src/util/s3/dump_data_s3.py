#%%
import boto3
import os
from pathlib import Path

#%%
# create S3 client
S3 = boto3.client('s3')
bucket_name = "nextiles-data-playground"

# S3 save location and name
src_directory = Path(".") / "src" / "util" / "dump_S3"
save_filename = "test-file.txt"
src_filename = src_directory / save_filename

#%%
# Upload
S3.upload_file(str(src_filename), bucket_name, save_filename)