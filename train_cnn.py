import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from incarcare import incarca_train
from validare import acuratete
from model_cnn import ReteaCNN

INALTIME = 128
LATIME = 55
NUMAR_CLASE = 5
BATCH = 128
BATCH_TEST = 384
EPOCI = 55 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FOLOSESC_AMP = torch.cuda.is_available()

translatie = T.Compose([T.RandomAffine(degrees=0, translate=(0.0, 0.05))])
cutout = T.RandomErasing(p=0.5, scale=(0.02, 0.08), ratio=(0.5, 2.0), value=0)

def pregateste_tensori(X, y=None, medie=0.0, dev_std=1.0):
    X = X.reshape(-1, 1, INALTIME, LATIME).astype(np.float32) / 255.0
    X = (X - medie) / (dev_std + 1e-8)
    X_t = torch.tensor(X)
    if y is not None:
        return X_t, torch.tensor((y - 1).astype(np.int64))
    return X_t

def mixup(x, y, intensitate=0.3):
    # Fuzionam 2 imagini din lot pentru a impiedica reteaua sa memoreze un exemplu si 
    # a o forta sa invete limite decizionale mai fine intre clase.
    raport = np.random.beta(intensitate, intensitate) if intensitate > 0 else 1
    idx = torch.randperm(x.size(0)).to(device)
    return raport * x + (1 - raport) * x[idx], y, y[idx], raport

def pierdere_mixup(criteriu, predictii, y_a, y_b, raport):
    return raport * criteriu(predictii, y_a) + (1 - raport) * criteriu(predictii, y_b)

def antreneaza(model, X_train, y_train, X_val, y_val):
    model = model.to(device)
    X_train, y_train, X_val = X_train.to(device), y_train.to(device), X_val.to(device)
    # Label smoothing-ul descurajeaza retelele sa devina exagerat de sigure pe o predictie
    criteriu = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-2)
    pasi_pe_epoca = math.ceil(len(X_train) / BATCH)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=4e-3, epochs=EPOCI, steps_per_epoch=pasi_pe_epoca, pct_start=0.2)
    scaler = torch.amp.GradScaler('cuda', enabled=FOLOSESC_AMP)

    cea_mai_buna_acc = 0.0
    cea_mai_buna_stare = None

    for epoca in range(EPOCI):
        model.train()
        permutare = torch.randperm(X_train.shape[0])
        for inceput in range(0, X_train.shape[0], BATCH):
            indici = permutare[inceput:inceput + BATCH]
            
            x_batch = translatie(X_train[indici])
            x_batch, y_a, y_b, raport = mixup(x_batch, y_train[indici], 0.3)
            x_batch = cutout(x_batch)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=FOLOSESC_AMP):
                iesire = model(x_batch)
                pierdere = pierdere_mixup(criteriu, iesire, y_a, y_b, raport)
                
            scaler.scale(pierdere).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        acc = acuratete(y_val, prezice(model, X_val))
        if acc > cea_mai_buna_acc:
            cea_mai_buna_acc = acc
            cea_mai_buna_stare = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(cea_mai_buna_stare)
    return cea_mai_buna_acc, cea_mai_buna_stare

def prezice(model, X):
    model.eval()
    X = X.to(device)
    rezultate = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_TEST):
            with torch.amp.autocast('cuda', enabled=FOLOSESC_AMP):
                iesire = model(X[i:i+BATCH_TEST]).float()
            rezultate.append(torch.argmax(iesire, dim=1).cpu().numpy())
    return np.concatenate(rezultate) + 1 

def inferenta_tta(model, X):
    #deplasam imaginea pe setul de test pentru robustete suplimentara.imaginea centrala si nealterata primeste ponderea dominanta
    model.eval()
    deplasari = [
        (0, 0, 3.0),
        (0, -1, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (1, 0, 1.0),
        (0, -2, 0.5), (0, 2, 0.5), (-2, 0, 0.5), (2, 0, 0.5)
    ]
    
    rezultat_ponderat = torch.zeros((len(X), NUMAR_CLASE)).to(device)
    suma_ponderi = 0.0
    
    with torch.no_grad():
        for dx, dy, pondere in deplasari:
            if dx != 0 or dy != 0:
                x_dep = TF.affine(X, angle=0, translate=[dx, dy], scale=1.0, shear=0)
            else:
                x_dep = X
            x_dep = x_dep.to(device)
            logiti = []
            
            for i in range(0, len(x_dep), BATCH_TEST):
                with torch.amp.autocast('cuda', enabled=FOLOSESC_AMP):
                    logiti.append(model(x_dep[i:i+BATCH_TEST]).float())
                    
            prob = F.softmax(torch.cat(logiti, dim=0), dim=1)
            rezultat_ponderat += prob * pondere
            suma_ponderi += pondere
            
    return (rezultat_ponderat / suma_ponderi).cpu().numpy()

def genereaza_si_salveaza_matrice(y_adevarat, y_prezis, cale_salvare='rezultate/confuzie_cnn.png'):
    os.makedirs(os.path.dirname(cale_salvare), exist_ok=True)
    cm = confusion_matrix(y_adevarat, y_prezis)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('CNN')
    plt.colorbar()
    
    clase = [1, 2, 3, 4, 5]
    tick_marks = np.arange(len(clase))
    plt.xticks(tick_marks, clase)
    plt.yticks(tick_marks, clase)
    plt.ylabel('Clasa reala')
    plt.xlabel('Clasa prezisa')
    
    prag = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > prag else "black")
    
    plt.tight_layout()
    plt.savefig(cale_salvare, dpi=300)
    plt.close()
    print(f'Matrice de confuzie salvata in {cale_salvare}')



def antreneaza_5fold(folder='data'):
    print('Incarc datele...')
    X_train, y_train = incarca_train(folder)

    numar_folduri = 5
    impartitor = StratifiedKFold(n_splits=numar_folduri, shuffle=True, random_state=42)

    modele = []
    statistici = []
    predictii_oof = np.zeros(len(y_train), dtype=int)

    for fold, (idx_tr, idx_va) in enumerate(impartitor.split(X_train, y_train)):
        print(f'\n-Fold {fold+1}/{numar_folduri}')
        medie = np.mean(X_train[idx_tr].astype(np.float32) / 255.0)
        dev_std = np.std(X_train[idx_tr].astype(np.float32) / 255.0)
        statistici.append((medie, dev_std))

        X_tr_t, y_tr_t = pregateste_tensori(X_train[idx_tr], y_train[idx_tr], medie, dev_std)
        X_va_t, _ = pregateste_tensori(X_train[idx_va], y_train[idx_va], medie, dev_std)

        model = ReteaCNN(numar_clase=NUMAR_CLASE)
        acc, stare = antreneaza(model, X_tr_t, y_tr_t, X_va_t, y_train[idx_va])
        modele.append(stare)

        model.load_state_dict(stare)
        prob_va = inferenta_tta(model, X_va_t)
        predictii_oof[idx_va] = np.argmax(prob_va, axis=1) + 1
        print(f'   acuratete fold: {acc:.4f}')

    print(f'\nAcuratete finala: {acuratete(y_train, predictii_oof):.4f}')
    genereaza_si_salveaza_matrice(y_train, predictii_oof, 'rezultate/confuzie_cnn.png')

    return modele, statistici

if __name__ == '__main__':
    antreneaza_5fold('data')