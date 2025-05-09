#Libraries for data loading and manipulation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#Libraries to preprocess and build model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.preprocessing import StandardScaler, OneHotEncoder

#Constants to give path for both datasets (artist data and image data)
#artist_fame pulled from https://github.com/Idanlau/Art_Price_Prediction_Model (they just added the fame category to the existing data frame)
CSV_PATH = "./artist_fame.csv"
IMG_PATH = "/Users/nicomonsalve/.cache/kagglehub/datasets/flkuhm/art-price-dataset/versions/1/artDataset"

#Class to prepare art dataset to be used by Pytorch
class ArtDataset(Dataset):
    
    #Constructor to initialize an instance of the ArtDataset class and preprocess data
    def __init__(self, df, encoders=None, scaler=None, transform=None):
        
        #Stores a copy of the dataframe (we don't want to manipulate the dataframe itself)
        self.df = df.copy()

        #Stores function used to transoform images
        self.transform = transform

        #Clean signed data (currently a message) by simplifying to a word
        self.df['signed_clean'] = self.df['signed'].fillna('none')
        self.df['signed_clean'] = self.df['signed'].fillna('none').str.lower()
        def clean_signed(s):
            if 'signed' in s:
                return 'signed'
            elif 'titled' in s:
                return 'titled'
            elif 'dated' in s:
                return 'dated'
            else:
                return 'other'
        
        #Clean condition data (currently a message) by simplifying to a word
        self.df['condition_clean'] = self.df['condition'].fillna('other')
        self.df['condition_clean'] = self.df['condition_clean'].str.lower()
        def clean_condition(c):
            if 'excellent' in c:
                return 'excellent'
            elif 'very good' in c:
                return 'very good'
            elif 'good' in c:
                return 'good'
            elif 'not examined' in c:
                return 'not examined'
            else:
                return 'other'
        
        self.df['signed_clean'] = self.df['signed_clean'].apply(clean_signed)
        self.df['condition_clean'] = self.df['condition_clean'].apply(clean_condition)
        
        #To convert/ encode categorical data (above) into a vector
        if encoders is None:
            #Three categorical data sets to encode
            self.encoders = {
                'period': OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(self.df[['period']]),
                'signed': OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(self.df[['signed_clean']]),
                'condition': OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(self.df[['condition_clean']])
            }
        else:
            self.encoders = encoders

        #Store encoded categorical variables in features array
        features = []
        encode_period = self.encoders['period'].transform(self.df[['period']])
        encode_signed = self.encoders['signed'].transform(self.df[['signed_clean']])
        encode_condition = self.encoders['condition'].transform(self.df[['condition_clean']])

        features.append(encode_period)
        features.append(encode_signed)
        features.append(encode_condition)

        #Takes fame and year column and fills missing values with 0 and converts to array
        fame = self.df[['fame']].fillna(0).values
        year = self.df[['yearCreation']].fillna(0).values

        #Concatenates the frame and year in order to scale them (StandardScaler only takes the values combined)
        raw_features = np.hstack([fame, year])

        #Scale our numeric values in order to normalize them between 0 and 1
        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = StandardScaler().fit(raw_features)
        
        numeric_features = self.scaler.transform(raw_features)

        #Create a single matrix of features
        self.features = np.hstack(features)

        #Get all the features
        self.numeric_features = numeric_features
        self.prices = self.df['price'].values
        self.image_ids = self.df['image_id'].astype(int).values

    #Method to return the length of the data frame through calling len()
    def __len__(self):
        return len(self.df)

    #Dataloader subclass
    def __getitem__(self, i):
        img_path = f"{IMG_PATH}/image_{self.image_ids[i] + 1}.png"
        img = Image.open(img_path).convert("RGB")

        #Transform the image
        if self.transform:
            img = self.transform(img)

        #Combine categorical values and numerical ones into a vecotr and set to float to not lose specificity
        x_vec = np.concatenate([self.features[i], self.numeric_features[i]])
        x_vec = torch.tensor(x_vec, dtype=torch.float32)
        y = torch.tensor(self.prices[i], dtype=torch.float32)
        return img, x_vec, y, f"image_{self.image_ids[i] + 1}.png"

#CNN logic
class ConvNet(nn.Module):
    def __init__(self, feature_dimension):
        super().__init__()
        
        #Convolutional layers which takes image with 3 channels (red, green, blue) and applies 16 3x3 filters to output 16 maps
        #Apply ReLU and shrinks image
        #Second layer takes the 16 outputs from first and applies similar process to get 32 outputs
        #Then flatten to convert into a 1d vector
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Flatten()
        )

        #Help us find size of output from cnn while we experiment with it
        with torch.no_grad():
            temp = torch.zeros(1, 3, 256, 256)
            conv_out_dimension = self.conv(temp).shape[1]

        #Feature layers run sequentially, get 64 patterns from features and then 32 from 64
        self.feature_net = nn.Sequential(
            nn.Linear(feature_dimension, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU()
        )

        #Reduces feature vector to 128, and then to 1 (our output)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_dimension + 32, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    #Passes image, features, and combines the vectors to return a prediction
    def forward(self, img, tab):
        x_img = self.conv(img)
        x_feat = self.feature_net(tab)
        x = torch.cat([x_img, x_feat], dim=1)
        return self.fc(x).squeeze()

def main():
    #Load in data
    df = pd.read_csv(CSV_PATH)

    #Clarify the name of the first column
    df.columns = df.columns.str.strip()
    df = df.rename(columns={'Unnamed: 0': 'image_id'})

    #Process the data
    df['yearCreation'] = pd.to_numeric(df['yearCreation'], errors='coerce')
    df['price'] = df['price'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False).str.replace(' USD', '', regex=False).astype(float)
    df = df.dropna(subset=['price', 'fame', 'period', 'image_id'])
    df = df.sort_values(by='image_id').reset_index(drop=True)
    df['price'] = np.log(df['price'])

    #Set in sample and out of sample datasets
    split_idx = int(0.9 * len(df))
    df_train, df_test = df.iloc[:split_idx], df.iloc[split_idx:]

    #Transofrm image by first resizing to 256x256 pixels, then converting image to tensor, then normalize rgb vals
    tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    #clean, normalize, and transform image
    train_ds = ArtDataset(df_train, transform=tfm)
    test_ds = ArtDataset(df_test, encoders=train_ds.encoders, scaler=train_ds.scaler, transform=tfm)

    #train shuffled data to reduce any bias (unsure if there is an ordering to the dataset)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=32)

    #Run model and choose cuda if available as emphasized previously on hws
    model = ConvNet(feature_dimension=len(train_ds[0][1]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #adapt learning rate and compute mse
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    #track if mae goes above best mae 5 times (to avoid overtraining)
    best_mae = float('inf')
    patience = 5
    wait = 0

    #track metrics
    train_losses = []
    test_maes = []

    #Run epochs
    for epoch in range(20):
        #Run training
        model.train()

        #Track total loss
        total = 0

        #Loop through training data
        for x_img, x_feat, y, _ in train_dl:
            #Move to correct device
            x_img = x_img.to(device)
            x_feat = x_feat.to(device)
            y = y.to(device)

            #Get prediciton through forward pass and loss
            pred = model(x_img, x_feat)
            loss = loss_fn(pred, y)

            #Reset gradient
            opt.zero_grad()

            #Get gradient of loss
            loss.backward()

            #update model weight
            opt.step()

            #Calculate total loss
            total += loss.item()
        
        train_loss = total / len(train_dl)

        #evaluate
        model.eval()
        total = 0

        #No trading so dont compute gradient 
        with torch.no_grad():

            #Iterate over dataset
            for x_img, x_feat, y, _ in test_dl:
                #Use correct device
                x_img = x_img.to(device)
                x_feat = x_feat.to(device)
                y = y.to(device)
                
                #Get prediction
                pred = model(x_img, x_feat)

                #Get dollar error for each prediction
                total += torch.mean(torch.abs(torch.exp(pred) - torch.exp(y))).item()
        
        test_mae = total / len(test_dl)

        train_losses.append(train_loss)
        test_maes.append(test_mae)
        
        #Print results
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Test MAE = ${test_mae:.2f}")

        #Break training if mae is greater than the best mae 5 times in a row
        if test_mae < best_mae:
            best_mae = test_mae
            best_weights = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience and epoch >= 10:
                print("MAE wait threshold was broken (stopping training early)")
                break
    
    #Load the best model as the one we use
    model.load_state_dict(best_weights)

    #Test out of sample appraisals given the trained model
    print("\n Out of sample predictions:")
    
    #Overall performance metric cache
    out_sample_avg_pct_off_mag = 0
    out_sample_avg_pct_off = 0
    decile_num = 0
    quartile_num = 0
    second_quartile = 0
    total_num_out_of_sample = 0

    #Evaluate on out of sample data
    model.eval()

    #Dont compute gradient
    with torch.no_grad():

        #Itterate through out of sample data
        for i in range(len(test_ds)):
            img, tab, y, fname = test_ds[i]

            #Get prediction
            pred = model(img.unsqueeze(0).to(device), tab.unsqueeze(0).to(device))

            #calculate performance
            pct_off = ((torch.exp(pred).item() - torch.exp(y).item()) / torch.exp(y).item()) * 100
            pct_off_mag = abs(pct_off)

            print(f"{fname} - Predicted: ${torch.exp(pred).item()}, Actual: ${torch.exp(y).item()}, Pct off: {pct_off}%")
            
            #Overall performance metrics
            out_sample_avg_pct_off_mag += pct_off_mag
            out_sample_avg_pct_off += pct_off

            #Check for number of artwork appraised within 10%
            if pct_off_mag <= 10:
                decile_num += 1
            
            #Check for number of artwork appraised within 25%
            if pct_off_mag <= 25:
                quartile_num += 1
            
            #Check for number of artwork appraised within 50%
            if pct_off_mag <= 50:
                second_quartile += 1

            total_num_out_of_sample += 1

    #Print of overall performance metrics 
    print("\n Performance metrics for out of sample artwork:")
    print(f"Average price missmatch: {out_sample_avg_pct_off / total_num_out_of_sample}%")       
    print(f"Average price missmatch magnitude: {out_sample_avg_pct_off_mag / total_num_out_of_sample}%")
    print(f"Percent of artwork appraised within 10% of true value: {(decile_num / total_num_out_of_sample) * 100}%")
    print(f"Percent of artwork appraised within 25% of true value: {(quartile_num / total_num_out_of_sample) * 100}%")
    print(f"Percent of artwork appraised within 50% of true value: {(second_quartile / total_num_out_of_sample) * 100}%")
    print(f"Percent of artwork appraised within outside of acceptible range (>50%): {(1- (second_quartile / total_num_out_of_sample)) * 100}%")

    #Plot the progress through learning to check that it seems ok (losses decreasing, mae relatively ok)
    train_losses = [loss * 100 for loss in train_losses]
    plt.plot(train_losses, label="train loss (log-mse) * 100")
    plt.plot(test_maes, label="test mean abs error ($)")
    plt.xlabel("epoch")
    plt.ylabel("loss and mean abs error")
    plt.title("training progress")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()