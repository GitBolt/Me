import torch
from model import NeuralNet
from prepare import tokenize_and_lemmatize, bag_of_words
import json


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


all_data = json.loads(open("conversations.txt", "r").read())

data = torch.load("data.pt", map_location='cuda')

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

while True:
    inp = input(">>")
    sentence = tokenize_and_lemmatize(inp)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    for data in all_data:
        if data["tag"] == tag:
            print(f"<< {data['reply']}")