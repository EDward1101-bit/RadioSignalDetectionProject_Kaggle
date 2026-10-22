import os
import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from incarcare import incarca_train

folder_rezultate = 'rezultate'

def diagrama_distributie_clase(etichete):

    numarari = np.bincount(etichete)

    clase = [1, 2, 3, 4, 5]
    valori = [numarari[c] for c in clase]

    plt.figure()
    plt.bar(clase, valori)
    plt.xlabel('Clasa (nr obiecte)')
    plt.ylabel('Numar de exemple ')
    plt.title("Distributia claselor in setul de antrenare")
    plt.savefig(os.path.join(folder_rezultate, 'distributie_clase.png'))
    plt.close()

    print('Distributie clase: ', valori)



def diagrama_exemple(imagini, etichete):
    plt.figure(figsize = (10, 4))

    for i, clasa in enumerate([1, 2, 3, 4, 5]):
        index = np.where(etichete == clasa)[0][0]#index ul primei imagini din clasa curenta

        imagine = imagini[index].reshape(128, 55)

        plt.subplot(1, 5, i + 1)
        plt.imshow(imagine, cmap='gray')
        plt.title('Clasa ' + str(clasa))
        plt.axis('off')

    plt.savefig(os.path.join(folder_rezultate, 'exemple_clase.png'))
    plt.close()


def diagrama_imagine_medie(imagini, etichete):
    plt.figure(figsize = (10, 4))

    for i, clasa in enumerate([1, 2, 3, 4, 5]):
        
        imagini_clasa = imagini[etichete == clasa]
        medie = imagini_clasa.mean(axis=0).reshape(128, 55)

        plt.subplot(1, 5, i + 1)
        plt.imshow(medie, cmap='gray')
        plt.title('Medie clasa ' + str(clasa))
        plt.axis('off')

    
    plt.savefig(os.path.join(folder_rezultate, 'imagine_medie.png'))
    plt.close()


if __name__ == '__main__':
    folder = 'data'
    imagini, etichete = incarca_train(folder)

    diagrama_distributie_clase(etichete)
    diagrama_exemple(imagini, etichete)
    diagrama_imagine_medie(imagini, etichete)

    print('Diagrame salvate in folder-ul: ', folder_rezultate)