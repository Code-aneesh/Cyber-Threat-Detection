import pandas as pd

# Feature names from datasets
unsw_features = ["dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes", "rate", 
                 "sttl", "dttl", "sload", "dload", "attack_cat", "label"]

cicids_features = ["Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets", 
                   "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Flow Bytes/s", 
                   "Flow Packets/s", "Fwd Packet Length Max", "Bwd Packet Length Max", "Label"]

# Remove 'label' columns and combine
combined_features = unsw_features[:-2] + cicids_features[:-1]

print("Total features in combined list:", len(combined_features))
print("Feature names:", combined_features)

# ✅ Fix selected indices (Removed index 24)
selected_indices = [2, 14, 15, 22]  # Choose only valid indices

# Extract feature names
selected_feature_names = [combined_features[i] for i in selected_indices if i < len(combined_features)]

print("📌 Selected Feature Names:", selected_feature_names)
