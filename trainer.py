import json
import numpy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet
from prepare import tokenize_and_lemmatize, bag_of_words

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
all_data = json.loads(open("conversations.txt", "r").read())

all_tokens = []
tags = []
xy = []

for data in all_data:
    tag = data['tag']
    tags.append(tag)
    tokenized_context = tokenize_and_lemmatize(data["context"])
    all_tokens.extend(tokenized_context)
    xy.append((tokenized_context, tag))
        
all_words = sorted(set(all_tokens)) 
tags = sorted(set(tags))

print(f"{len(xy)} Patterns\n{len(tags)} Tags: {tags}")
print("-"*100, f"\nTokens: {all_tokens}")

x_train = numpy.array(
    [bag_of_words(context, all_words) for (context, _) in xy])
y_train = numpy.array(
    [tags.index(tag) for (_, tag) in xy])

# Hyper-parameters 
num_epochs = 10
batch_size = 8
learning_rate = 0.05
input_size = len(x_train[0])
hidden_size = 8
output_size = len(tags)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

train_loader = DataLoader(dataset=ChatDataset(),
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)


model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("Started training...")
# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.2f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pt"
torch.save(data, FILE)

print(f'File saved to {FILE}')
