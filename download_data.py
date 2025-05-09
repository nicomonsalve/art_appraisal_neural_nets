#Pulled from kaggle developer documentation. Downloads the necessary data
import kagglehub

# Download latest version
path = kagglehub.dataset_download("flkuhm/art-price-dataset")

print("Path to dataset files:", path)