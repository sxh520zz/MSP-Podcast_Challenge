import pickle
import csv

# Define the emotion mapping
emotion_mapping = {
    "Angry": "A",
    "Sad": "S",
    "Happy": "H",
    "Surprise": "U",
    "Fear": "F",
    "Disgust": "D",
    "Contempt": "C",
    "Neutral": "N"
}

# Load the pickle file
pickle_file = '/home/shixiaohan-toda/Desktop/Challenge/SHI/Baseline_Whisper/Final_result_test.pickle'
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

# Prepare the data for CSV
csv_data = []
for item in data[0]:
    id = item['id']
    label = item['Predict_label']
    
    # Map the label to the corresponding emotion class
    emotion_class = list(emotion_mapping.keys())[label]
    emo_class = emotion_mapping[emotion_class]
    
    # Prepare the row with filename and emotion class
    filename = f"{id}.wav"
    csv_data.append([filename, emo_class])

# Write the data to a CSV file
csv_file = '/home/shixiaohan-toda/Desktop/Challenge/SHI/Baseline_Whisper/output_emotion_classes.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['FileName', 'EmoClass'])
    writer.writerows(csv_data)

print(f"CSV file saved as {csv_file}")