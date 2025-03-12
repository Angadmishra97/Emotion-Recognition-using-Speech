import pandas as pd

data = {
    "filename": ["03-01-07-01-02-02-24.wav", "03-01-07-02-01-01-24.wav", "03-01-07-02-01-02-24.wav"],
    "label": ["Angry", "Happy", "Neutral"],
}

df = pd.DataFrame(data)

# Save CSV file
df.to_csv("labels.csv", index=False)

print("CSV file created successfully!")
