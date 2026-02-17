from flask import Flask, render_template, request
import torch
import torch.nn as nn
import pickle

# -----------------------
# Load vocab
# -----------------------
with open("vocab.pkl", "rb") as f:
    word2idx = pickle.load(f)

vocab_size = len(word2idx)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# SimpleBERT (MUST match training)
# -----------------------
class SimpleBERT(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size=128,
        max_len=128,
        num_layers=2,
        num_heads=4,
        dropout=0.1
    ):
        super(SimpleBERT, self).__init__()

        self.hidden_size = hidden_size
        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_len, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, return_hidden=False):

        batch_size, seq_len = input_ids.size()

        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)

        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)

        x = token_embeddings + position_embeddings
        x = self.dropout(x)

        x = self.encoder(x)

        if return_hidden:
            return x

        return x


# -----------------------
# SentenceBERT 
# -----------------------
class SentenceBERT(nn.Module):
    def __init__(self, bert_model):
        super(SentenceBERT, self).__init__()
        self.bert = bert_model
        hidden_size = bert_model.hidden_size
        self.classifier = nn.Linear(hidden_size * 3, 3)

    def mean_pool(self, token_embeddings):
        return torch.mean(token_embeddings, dim=1)

    def forward(self, input_ids1, input_ids2):

        out1 = self.bert(input_ids1, return_hidden=True)
        out2 = self.bert(input_ids2, return_hidden=True)

        emb1 = self.mean_pool(out1)
        emb2 = self.mean_pool(out2)

        diff = torch.abs(emb1 - emb2)
        features = torch.cat([emb1, emb2, diff], dim=1)

        logits = self.classifier(features)
        return logits


# -----------------------
# Create Model
# -----------------------
bert_model = SimpleBERT(
    vocab_size=vocab_size,
    hidden_size=128,
    max_len=128,   # MUST match training
    num_layers=2,
    num_heads=4
).to(device)

model = SentenceBERT(bert_model).to(device)

model.load_state_dict(torch.load("sbert_model.pth", map_location=device), strict=False)
model.eval()

# -----------------------
# Tokenizer (must match training max_len=128)
# -----------------------
def tokenize(sentence, max_len=128):
    tokens = sentence.lower().split()

    ids = []
    for token in tokens:
        ids.append(word2idx.get(token, word2idx["<unk>"]))

    if len(ids) < max_len:
        ids += [word2idx["<pad>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    return torch.tensor([ids], dtype=torch.long).to(device)


# -----------------------
# Flask App
# -----------------------
app = Flask(__name__)

labels_map = {
    0: "Entailment",
    1: "Neutral",
    2: "Contradiction"
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        premise = request.form["premise"]
        hypothesis = request.form["hypothesis"]

        s1 = tokenize(premise)
        s2 = tokenize(hypothesis)

        with torch.no_grad():
            logits = model(s1, s2)
            pred = torch.argmax(logits, dim=1).item()
            prediction = labels_map[pred]

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
