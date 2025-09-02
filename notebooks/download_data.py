import requests
import zipfile
import os

# Download dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"
response = requests.get(url)

# Save and extract
os.makedirs("../data", exist_ok=True)
with open("../data/dataset_diabetes.zip", "wb") as f:
    f.write(response.content)

with zipfile.ZipFile("../data/dataset_diabetes.zip", "r") as zip_ref:
    zip_ref.extractall("../data")

print("Dataset downloaded and extracted to data/")