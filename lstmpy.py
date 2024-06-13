import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from transformers import AutoTokenizer

BATCH_SIZE = 100
tokenizer = AutoTokenizer.from_pretrained("projecte-aina/FLOR-6.3B")

#test
tokens = tokenizer.tokenize("hola que tal")
print(tokens)

class Dataset(Dataset):
    def __init__(self, csv, tokenizer):
        df = pd.read_csv(csv)
        self.x = df["review"].values
        self.y = df["sentiment"].values
        self.tokenizer = tokenizer
        y_encoded = pd.get_dummies(self.y)
        self.y_encoded_tensor = torch.tensor(y_encoded.values, dtype=torch.int32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        sentence = self.x[idx]
        tokens = self.tokenizer.tokenize(sentence)
        print(len(tokens))
        logits = self.y_encoded_tensor[idx]
        return tokens, logits

training_data = Dataset(csv="IMDB Dataset.csv", tokenizer=tokenizer)
test_data = Dataset(csv="IMDB Dataset.csv", tokenizer=tokenizer)

train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

#define hyperparameters
SEQUENCE_LEN = 28
INPUT_LEN = 28
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 1
NUM_EPOCHS = 5
LEARNING_RATE = 0.01

class LSTM(nn.Module):
    def __init__(self, input_len, hidden_size, num_class, num_layers):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_len, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, NUM_CLASSES)

    def forward(self, X):
        hidden_states = torch.zeros(self.num_layers, X.size(0), self.hidden_size)
        cell_states = torch.zeros(self.num_layers, X.size(0), self.hidden_size)
        #"_" --> Hidden state
        out, _ = self.lstm(X, (hidden_states, cell_states))
        out = self.output_layer(out[:, -1, :])
        return out
    
model = LSTM(INPUT_LEN, HIDDEN_SIZE, NUM_CLASSES, NUM_LAYERS)
print(model)

loss_func = nn.CrossEntropyLoss()
sgd_optim = optim.SGD(model.parameters(), lr=LEARNING_RATE)
adam_optim = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train(num_epochs, model, train_dataloader, loss_func, optimizer):
    total_steps = len(train_dataloader)

    for epoch in range(num_epochs):

        for batch, (text, labels) in enumerate(train_dataloader):
            output = model(text)
            loss = loss_func(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if(batch+1)%100 == 0:
                print(f"Epoch: {epoch}; Batch: {batch+1} /{total_steps}; Loss: {loss.item():>4f}")

train(NUM_EPOCHS, model, train_dataloader, loss_func, adam_optim)