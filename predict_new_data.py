from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch import torch

# Define the path to the saved model and tokenizer
model_path = r'C:/Users/Payso/ai-doc-agent/results/checkpoint-1500'

# Load the tokenizer and model from local files
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)

# Define your new data (for example, a single article)
new_data = [
    "Example news article 1: Grateful Dog Plants Slobbery Kisses On Rescuer Who Saved Him.  Duke, a Belgian Malinois and his owner Edward Emmerich, who are homeless and have been living in an encampment, found themselves trapped in floodwaters last Friday in McKinney, Texas, CBS DFW reported.  The duo was brought to safety by first responders and the rescuers' work definitely did not go unnoticed. Duke expressed his thanks in the best way a grateful dog can.  His feet hit the ground and he almost instantly went towards the firefighter that had saved him, jumped all over him, licked all over him, photographer Michael O'Keefe, who witnessed the scene, told NBC DFW. It was real touching to watch.",
    "Example news article 2: 9-Year-Old Honors Late Father In Powerful Remembrance Photos.  Photographer April Reeves helped 9-year-old Ethan pay tribute to his late father, Louisiana State Trooper Steven Vincent, by taking pictures of him with some of his dad's belongings. While editing the photos, Reeves also incorporated the fallen officer's image.  Senior Trooper Vincent died on the job in August after the suspect in a traffic investigation shot him. Colleagues described him as an honorable officer and all-around good guy, KPLC reports. After his passing, Vincent's wife Katherine contacted Reeves about taking some remembrance photos of Ethan with his father's trooper hat and flag. This was my first time doing a remembrance shoot, but I knew that I had to make it special for them, the photographer told The Huffington Post."
]

# Tokenize the new data
inputs = tokenizer(new_data, padding=True, truncation=True, return_tensors="pt")

# Run inference on the new data
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

# Print predictions for each text
for text, prediction in zip(new_data, predictions):
    print(f"Text: {text}")
    print(f"Predicted Label: {prediction.item()}")



