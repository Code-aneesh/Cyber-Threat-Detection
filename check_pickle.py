import joblib

# Load the pickle file
data = joblib.load("data/processed/selected_data.pkl")

# Print the type and content
print(f"Type of data: {type(data)}")

# If it's a tuple, print its length
if isinstance(data, tuple):
    print(f"Tuple Length: {len(data)}")
    for i, item in enumerate(data):
        print(f"Element {i} type: {type(item)}, shape: {getattr(item, 'shape', 'N/A')}")

# If it's a dictionary, print the keys
elif isinstance(data, dict):
    print(f"Dictionary Keys: {list(data.keys())}")
