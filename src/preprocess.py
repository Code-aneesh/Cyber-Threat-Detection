import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# -----------------------
# Load UNSW-NB15 Dataset
# -----------------------
print(" Loading UNSW-NB15 dataset...")
unsw_path = "C:/Users/user/OneDrive/Desktop/projects/cyber/data/raw/UNSW_NB15/UNSW_NB15_training-set.csv"

try:
    unsw_df = pd.read_csv(unsw_path)
    print(" UNSW-NB15 loaded successfully!")
except Exception as e:
    print(f" Error loading UNSW-NB15: {e}")
    unsw_df = None

# -----------------------
# Load CICIDS2017 Dataset
# -----------------------
print("\n Loading CICIDS2017 dataset...")
cicids_files = [
    "C:/Users/user/OneDrive/Desktop/projects/cyber/data/raw/CICIDS2017/Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv",
    "C:/Users/user/OneDrive/Desktop/projects/cyber/data/raw/CICIDS2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "C:/Users/user/OneDrive/Desktop/projects/cyber/data/raw/CICIDS2017/Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "C:/Users/user/OneDrive/Desktop/projects/cyber/data/raw/CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv",
    "C:/Users/user/OneDrive/Desktop/projects/cyber/data/raw/CICIDS2017/Thursday-WorkingHours-Afternoon-Infiltration.pcap_ISCX.csv",
    "C:/Users/user/OneDrive/Desktop/projects/cyber/data/raw/CICIDS2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "C:/Users/user/OneDrive/Desktop/projects/cyber/data/raw/CICIDS2017/Tuesday-WorkingHours.pcap_ISCX.csv",
    "C:/Users/user/OneDrive/Desktop/projects/cyber/data/raw/CICIDS2017/Wednesday-workingHours.pcap_ISCX.csv"
]

cicids_df = None

try:
    existing_files = [file for file in cicids_files if os.path.exists(file)]
    
    if existing_files:
        cicids_df = pd.concat([pd.read_csv(file) for file in existing_files], ignore_index=True)
        print(" CICIDS2017 loaded successfully!")
        
        # Standardize column names (remove leading/trailing spaces)
        cicids_df.columns = cicids_df.columns.str.strip()
        print("CICIDS2017 columns standardized.")
    else:
        print(" No CICIDS2017 files found. Skipping...")
except Exception as e:
    print(f" Error loading CICIDS2017: {e}")

# -----------------------
# Handle Missing Values (Fixed)
# -----------------------
if unsw_df is not None:
    print("\n Checking Missing Values in UNSW-NB15...")
    missing_unsw = unsw_df.isnull().sum().sum()
    print(f" {missing_unsw} missing values found.")
    
    if missing_unsw > 0:
        unsw_df.ffill(inplace=True)  # ✅ Fixed ffill warning

if cicids_df is not None:
    print("\n Checking Missing Values in CICIDS2017...")
    missing_cicids = cicids_df.isnull().sum().sum()
    print(f" {missing_cicids} missing values found.")
    
    if missing_cicids > 0:
        cicids_df.ffill(inplace=True)  # ✅ Fixed ffill warning

# -----------------------
# Select Relevant Features
# -----------------------
if unsw_df is not None:
    print("\n Selecting relevant features for UNSW-NB15...")
    unsw_features = [
        'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes', 
        'rate', 'sttl', 'dttl', 'sload', 'dload', 'attack_cat', 'label'
    ]
    
    existing_features_unsw = [feature for feature in unsw_features if feature in unsw_df.columns]
    unsw_df = unsw_df[existing_features_unsw]
    print(" UNSW-NB15 features selected.")

if cicids_df is not None:
    print("\n Selecting relevant features for CICIDS2017...")
    cicids_features = [
        'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 
        'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Flow Bytes/s', 
        'Flow Packets/s', 'Fwd Packet Length Max', 'Bwd Packet Length Max', 'Label'
    ]
    
    existing_features_cicids = [feature for feature in cicids_features if feature in cicids_df.columns]
    
    if existing_features_cicids:
        cicids_df = cicids_df[existing_features_cicids]
        print(" CICIDS2017 features selected.")
    else:
        print(" No matching features found in CICIDS2017 dataset.")

# -----------------------
# Convert Attack Labels to Numbers
# -----------------------
encoder = LabelEncoder()

if unsw_df is not None:
    print("\n Encoding labels in UNSW-NB15...")
    unsw_df['attack_cat'] = encoder.fit_transform(unsw_df['attack_cat'])
    print(" UNSW-NB15 labels encoded.")

if cicids_df is not None:
    print("\n Encoding labels in CICIDS2017...")
    cicids_df['Label'] = encoder.fit_transform(cicids_df['Label'])
    print(" CICIDS2017 labels encoded.")

# -----------------------
# Save Cleaned Data
# -----------------------
output_dir = "C:/Users/user/OneDrive/Desktop/projects/cyber/data/cleaned"
os.makedirs(output_dir, exist_ok=True)

if unsw_df is not None:
    print("\n Saving UNSW-NB15 cleaned data...")
    unsw_df.to_csv(f"{output_dir}/UNSW_NB15_clean.csv", index=False)
    print(" UNSW-NB15 cleaned dataset saved.")

if cicids_df is not None:
    print("\n Saving CICIDS2017 cleaned data...")
    cicids_df.to_csv(f"{output_dir}/CICIDS2017_clean.csv", index=False)
    print(" CICIDS2017 cleaned dataset saved.")

print("\n Preprocessing Complete!")
input("\nPress Enter to exit...")
