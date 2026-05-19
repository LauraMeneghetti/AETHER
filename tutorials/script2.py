import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import os
import sys
notebook_dir = os.getcwd()
parent_dir = os.path.join(notebook_dir, '..')
sys.path.append(parent_dir)

from aether.utils_rnn import find_best_threshold, calculate_metrics

X = np.load('data_window_10s.npy')
y = np.load('label_window_10s.npy')
y1 = np.load('label_window_1_10s.npy')
print(X[:, :, 0].shape)
#X = X[:10000, :, 0]
X = X[:, :, 0]
print(X.shape)
#y = y[:10000]
print(y.shape)

#counting the occurrence of a certain class in y
unique, counts = np.unique(y, return_counts=True)
D=dict(zip(unique, counts))
import matplotlib.pyplot as plt

vals_dict = {}
for i in y:
    if i in vals_dict.keys():
        vals_dict[i] += 1
    else:
        vals_dict[i] = 1
total = sum(vals_dict.values())
print(vals_dict)
# Formula used - Naive method where
# weight = 1 - (no. of samples present / total no. of samples)
# So more the samples, lower the weight
weight_dict = {int(k): (1 - (v / total)) for k, v in vals_dict.items()}

print(weight_dict)

vals_dict1 = {}
for i in y1:
    if i in vals_dict1.keys():
        vals_dict1[i] += 1
    else:
        vals_dict1[i] = 1
total = sum(vals_dict1.values())
print(vals_dict1)

# Formula used - Naive method where
# weight = 1 - (no. of samples present / total no. of samples)
# So more the samples, lower the weight
weight_dict1 = {int(k): (1 - (v / total)) for k, v in vals_dict1.items()}

print(weight_dict1)

x_train, x_test, y_train, y_test =train_test_split(
    X, y, test_size=0.2, #shuffle=True #random_state=42, shuffle=True
)
#y_train = np.asarray(y_train).astype(np.float32).reshape(-1, 1)

#y_test = np.asarray(y_test).astype(np.float32).reshape(-1, 1)
x_train = torch.tensor(x_train, dtype=torch.float)
x_test = torch.tensor(x_test, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)
print('Test size', x_test.size())
print(y_test.size())


import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, d_model, head_size, num_heads, ff_dim, dropout):
        super().__init__()
        # In PyTorch embed_dim = d_model
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Conv1d(d_model, ff_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(ff_dim, d_model, kernel_size=1)
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Normalization + Attention + Residual
        res = x
        x = self.layernorm1(x)
        attn_out, _ = self.mha(x, x, x)
        x = res + self.dropout1(attn_out)

        # Feed Forward + Residual
        res = x
        x = self.layernorm2(x)
        x = x.transpose(1, 2)
        x = self.ffn(x)
        x = x.transpose(1, 2)
        return res + self.dropout2(x)

class CustomAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Architettura basata sul paper (132 -> 64 -> 1)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 132, kernel_size=7, padding=3, stride=2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(132, 64, kernel_size=7, padding=3, stride=2),
            nn.ReLU()
        )

        # Upsampling per tornare alla lunghezza originale
        self.decoder = nn.ConvTranspose1d(64, 1, kernel_size=7, padding=3, stride=4, output_padding=3)

    def forward(self, x):
        # x: (batch, 1, seq_len)
        return self.decoder(self.encoder(x))

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Primo strato: da 1 canale SpO2 a 132 feature
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=132, kernel_size=7, padding=3, stride=2)
        self.dropout = nn.Dropout(p=0.1)

        # SECONDO STRATO (Qui era l'errore): deve accettare 132 canali!
        self.conv2 = nn.Conv1d(in_channels=132, out_channels=64, kernel_size=7, padding=3, stride=2)

        # Strato di ricostruzione (Decoder)
        # Riporta a 1 canale per sommarlo all'input originale come Positional Encoding
        self.decoder = nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size=7, padding=3, stride=4, output_padding=3)

    def forward(self, x):
        # x shape: (batch, 1, 80)
        #print('x', x.size())
        x_enc = F.relu(self.conv1(x))
        x_enc = self.dropout(x_enc)
        x_enc = F.relu(self.conv2(x_enc))

        # Ricostruzione
        pos_encoding = self.decoder(x_enc)
        return pos_encoding

class ApneaTransformer(nn.Module):
    def __init__(self, n_classes, head_size, num_heads, ff_dim, num_blocks, mlp_units, dropout, mlp_dropout):
        super().__init__()

        # 1. AutoEncoder (Generatore di Positional Embedding)
        self.auto_pe = AutoEncoder()

        # 2. Input Projection
        # Trasformiamo l'input (1 canale) nella dimensione d_model del Transformer
        self.d_model = head_size * num_heads
        self.input_projection = nn.Linear(1, self.d_model)

        # 3. Transformer Encoder Blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.d_model, head_size, num_heads, ff_dim, dropout)
            for _ in range(num_blocks)
        ])

        # 4. MLP Head
        layers = []
        curr_in = self.d_model
        for dim in mlp_units:
            layers.append(nn.Linear(curr_in, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(mlp_dropout))
            curr_in = dim

        self.mlp_head = nn.Sequential(*layers)
        self.classifier = nn.Linear(curr_in, n_classes)

    def forward(self, x):
        # x: (batch, 1, seq_len)

        # LOGICA PAPER: Sommiamo l'input originale al PE generato dall'AutoEncoder
        pe = self.auto_pe(x)
        x_combined = x + pe # Questa è la somma data-driven

        # Trasponiamo per il Transformer: (batch, seq, feat)
        x_combined = x_combined.transpose(1, 2)
        x_projected = self.input_projection(x_combined)

        # Blocchi Transformer
        for block in self.transformer_blocks:
            x_projected = block(x_projected)

        # Global Average Pooling (Lungo la dimensione temporale)
        #x_pooled = x_projected.mean(dim=1)
        # x_projected ha shape: (batch, seq_len, d_model)
        # Facciamo il max lungo la dimensione temporale (dim=1) per catturare il picco di desaturazione
        x_pooled, _ = x_projected.max(dim=1)

        # MLP Head
        x_mlp = self.mlp_head(x_pooled)
        return self.classifier(x_mlp)
    

device='cuda'
model = ApneaTransformer(
    n_classes=2,
    head_size=32, #256,
    num_heads=4,
    ff_dim=128, #4,
    num_blocks=4, #6,
    mlp_units=[128], #[256],
    dropout=0.2, #0.1,
    mlp_dropout=0.3 #0.2
).to(device)


import torch.optim as optim
device = 'cuda'
# Parametri presi dal tuo codice Keras
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
# weight_dict deve essere un tensore con i pesi calcolati
weights = torch.tensor([weight_dict[0], weight_dict[1]], dtype=torch.float).to(device)
criterion = torch.nn.CrossEntropyLoss() #weight=weights)
import torch
from torch.utils.data import DataLoader, TensorDataset


class_counts = np.bincount(y_train)  #coutn occurences of each class
class_weights = 1./ class_counts
weights = class_weights[y_train]  #assign weight to each sample

#create a sampler using these weights (imbalances dataset)
sampler = WeightedRandomSampler(weights, num_samples=len(y_train), replacement=True)



# Prepariamo i dati
train_ds = TensorDataset(
    torch.tensor(x_train).float(),
    torch.tensor(y_train).long() # <-- Corretto se y_train è 1D
)
train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler) # shuffle=True, sampler=sampler)

# Scheduler (equivalente alla tua funzione scheduler)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9048) # exp(-0.1) ≈ 0.9048

best_val_loss = float('inf')
patience = 10
trigger_times = 0

#state_dict = torch.load("autoenc_50ep_fulldata_10s_newparam.pth", weights_only=True)
#model.load_state_dict(state_dict)


from sklearn.metrics import f1_score, roc_auc_score

def evaluate_model(model, x_test, y_test, device):
    model.eval()
    # Prepariamo il loader di test
    test_ds = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.unsqueeze(1).to(device)
            outputs = model(batch_x)

            # Applichiamo Softmax per le probabilità della classe 1 (Apnea)
            probs = F.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(batch_y.numpy())

    # Calcolo metriche robuste per sbilanciamento
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = 0.5 # Se il modello predice una sola classe costantemente

    return f1, auc




for epoch in range(150):
    print('Epoch', epoch)
    model.train()
    epoch_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.unsqueeze(1)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Step dello scheduler dopo l'epoca 10 (come nel tuo codice)
    if epoch >= 10:
        scheduler.step()
    #print('loss', loss.item()/len(train_loader))
    print(f'Train Loss Media: {epoch_loss / len(train_loader):.4f}')

    val_f1, val_auc = evaluate_model(model, x_test, y_test, device)
    print(f"Validation Metrics -> F1-Score: {val_f1:.4f} | AUC-ROC: {val_auc:.4f}")

    # # --- LOGICA VALIDAZIONE & EARLY STOPPING ---
    # model.eval()
    # val_loss = calculate_val_loss() # Funzione helper da definire

    # if val_loss < best_val_loss:
    #     best_val_loss = val_loss
    #     torch.save(model.state_dict(), "training_2/cp.ckpt")
    #     trigger_times = 0
    # else:
    #     trigger_times += 1
    #     if trigger_times >= patience:
    #         print("Early stopping!")
    #         break

torch.save(model.state_dict(), "autoenc_150ep_fulldata_10s_script2.pth")

#state_dict = torch.load("autoenc_50ep_fulldata.pth", weights_only=True)
#model.load_state_dict(state_dict)

from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC

# Inizializzazione
metric_acc = BinaryAccuracy().to(device)
metric_f1 = BinaryF1Score().to(device)
metric_auc = BinaryAUROC().to(device)

# Durante la valutazione:
#with torch.no_grad():
#    x_test = x_test.unsqueeze(1)
#    preds = model(x_test.to(device))
#    # Calcolo
#    acc = metric_acc(preds.argmax(dim=1), y_test)
#    f1 = metric_f1(preds.argmax(dim=1), y_test)
#    auc = metric_auc(preds[:, 1], y_test)

#print(acc)
#print(f1)
#print(auc)


import torch
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection, Accuracy, F1Score, AUROC, Precision, Recall, Specificity

# Inizializziamo le metriche (assumendo classificazione binaria come nel tuo esempio)
metrics = MetricCollection({
    'accuracy': Accuracy(task="binary"),
    'f1_score': F1Score(task="binary"),
    'auc': AUROC(task="binary"),
    'precision': Precision(task="binary"),
    'recall': Recall(task="binary"),
    'specificity': Specificity(task="binary")
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metrics.to(device)


test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64)

def evaluate_pytorch(model, test_loader, criterion):
    model.eval() # Imposta il modello in modalità valutazione
    metrics.reset()
    running_loss = 0.0

    with torch.no_grad(): # Disabilita il calcolo dei gradienti (risparmia memoria)
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_batch = x_batch.unsqueeze(1)
            # Forward pass
            outputs = model(x_batch)

            # Se il modello restituisce logits, applichiamo sigmoid per le metriche
            #probs = torch.sigmoid(outputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            # Calcolo Loss
            loss = criterion(outputs, y_batch.long())
            running_loss += loss.item()

            # Aggiornamento metriche
            metrics.update(probs, y_batch.int())

    # Calcolo finale
    total_loss = running_loss / len(test_loader)
    final_metrics = metrics.compute()
    
    print(f"Loss: {total_loss:.4f}")
    for key, value in final_metrics.items():
        print(f"{key}: {value.item():.4f}")

# Esecuzione
criterion = torch.nn.CrossEntropyLoss()
evaluate_pytorch(model.to(device), test_loader, criterion)






def evaluate_model(model, data_loader, device):
    """
    Function performing the model's forward pass to gain the predictions,
    class probabilities and ordered real labels.

    :param nn.Module model: input model
    :param DataLoader data_loader: data loader we are considering for evaluation
    :param torch.device device: device

    :return: accurcay of the model, predictions, related class probabilities and 
             ordered labels
    :rtype: float, np.array, np.array, np.array
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            X_batch = X_batch.unsqueeze(1) # (batch, seq_len) -> (batch, seq_len, 1)

            outputs = model(X_batch)
            outs = torch.softmax(outputs, dim=1)[:, 1]
            predictions = (outs > 0.5).long()

            all_preds.append(predictions.cpu())
            all_probs.append(outs.cpu())
            all_labels.append(y_batch.cpu())

    all_preds_np = torch.cat(all_preds, dim=0).numpy()
    all_probs_np = torch.cat(all_probs, dim=0).numpy()
    all_labels_np = torch.cat(all_labels, dim=0).numpy() 

    acc = accuracy_score(all_labels_np, all_preds_np)

    return acc, all_preds_np, all_labels_np, all_probs_np




val_accuracy, y_pred, y_true, y_probs = evaluate_model(model, test_loader, device)
print(f"Validation Accuracy: {val_accuracy:.2f}")

lstm_probs_val = np.array(y_probs)

best_thresh = find_best_threshold(np.array(y_true), lstm_probs_val, method='f1')
print(f"Optimal threshold: {best_thresh:.4f}")

# Apply threshold to new predictions
y_pred = (lstm_probs_val >= best_thresh).astype(int)

metrics_dict = calculate_metrics(y_true, y_pred, lstm_probs_val)
