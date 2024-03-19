import streamlit as st
import torch
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from torch.utils.data import Dataset, DataLoader
import random
import json

nltk.download("punkt")


class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, hidden_size)
        self.l3 = torch.nn.Linear(hidden_size, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out


class ChatDataset(Dataset):
    def __init__(self, X_train, y_train):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag


# Load data
intents_file_path = r"C:\Users\Anuj Bohra\Desktop\chatbot\data\intents.json"

with open(intents_file_path, "r") as f:
    intents = json.load(f)

stemmer = PorterStemmer()

all_words = []
tags = []
xy = []

for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ["?", ".", "!"]
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []

for pattern_sentence, tag in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Model parameters
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000
batch_size = 8

# Initialize model, dataset, dataloader
model = NeuralNet(input_size, hidden_size, output_size)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
dataset = ChatDataset(X_train, y_train)
train_loader = DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0
)

# Load PyTorch model
checkpoint = torch.load(r"C:\Users\Anuj Bohra\Desktop\chatbot\data.pth")
model.load_state_dict(checkpoint["model_state"])
model.eval()

# Streamlit UI
st.title("Medical ChatBot")

# Set page layout with custom CSS
st.markdown(
    """
    <style>
    body {
        color: #333333;
        background-color: #ffffff;
        font-size: 16px;
    }
    .reportview-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 100%;
    }
    .btn {
        margin-top: 10px;
        margin-bottom: 10px;
        padding-left: 20px;
        padding-right: 20px;
        padding-top: 10px;
        padding-bottom: 10px;
        border-radius: 5px;
        background-color: #3366ff;
        color: white;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease 0s;
    }
    .btn:hover {
        background-color: #004080;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

user_input = st.text_input("You:", "")

if st.button("Ask"):
    sentence = user_input
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                response = random.choice(intent["responses"])
                st.text("Bot: " + response)
    else:
        st.text("Bot: I do not understand...")
