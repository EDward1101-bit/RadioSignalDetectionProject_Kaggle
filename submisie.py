import os
import numpy as np
import pandas as pd

from incarcare import incarca_test
from model_cnn import ReteaCNN
from train_cnn import antreneaza_5fold, pregateste_tensori, inferenta_tta, device, NUMAR_CLASE

def genereaza_submisie(folder='data', fisier_iesire='rezultate/submisie.csv'):
    
    modele, statistici = antreneaza_5fold(folder)

    print('\nIncarc datele de test...')
    X_test, id_test = incarca_test(folder)

    print('Generez predictiile...')
    prob_totala = np.zeros((len(X_test), NUMAR_CLASE))
    model_inferenta = ReteaCNN(numar_clase=NUMAR_CLASE).to(device)


    # iteram prin toate modelele antrenate 
    # Insumam prob. la iesire pentru a asigura rezultate stabile
    for i in range(len(modele)):
        medie, dev_std = statistici[i]
        X_test_t = pregateste_tensori(X_test, None, medie, dev_std)
        
        model_inferenta.load_state_dict(modele[i])
        prob_totala += inferenta_tta(model_inferenta, X_test_t)

    prob_totala /= len(modele)
    predictii = np.argmax(prob_totala, axis=1) + 1  
    
    os.makedirs(os.path.dirname(fisier_iesire), exist_ok= True)
    tabel = pd.DataFrame({'id': id_test, 'label': predictii})
    tabel.to_csv(fisier_iesire, index=False)
    
    print(f'\nSubmisie salvata in {fisier_iesire}')
    print('Distributie clase prezise:', dict(zip(*np.unique(predictii, return_counts = True))))

if __name__ == '__main__':
    # Submisia generata e doar pentru modelul CNN, pentru rezultatele celorlalte modele se foloseste main-ul
    genereaza_submisie('data', 'rezultate/submisie.csv')