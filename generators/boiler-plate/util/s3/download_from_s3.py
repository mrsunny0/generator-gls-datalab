#%%
import boto3, csv
from pathlib import Path

#%%
bucket = "BUCKET"

#%%
def download_csv_from_bucket(bucket_name, save_dir=Path.cwd()):
    all_objects = s3.list_objects(Bucket=bucket_name)['Contents']
    
    for obj in all_objects:
        # get object full S3 path and base name
        obj_path = obj['Key']
        obj_name = Path(obj_path).name
        
        # if file is .csv, then download and save locally
        if Path(obj_path).suffix == '.csv':
            # append base name to save_dir for save_path 
            save_path = save_dir / obj_name
            
            # save locally into save_path
            s3.download_file(
                bucket_name,
                obj_path,
                str(save_path))

#%%
s3 = boto3.client('s3')
download_path = Path.cwd() / "data" / "raw"
for bucket in buckets:
    download_csv_from_bucket(bucket, download_path)