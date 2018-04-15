<h1 align='center'>Deep learning <p> Art styles recognition
 </h1>
<p align='center'>
<i>Option ISIA - Centrale Paris <br>
Projet Deep learning <br>
Mars - Avril 2018 <hr></i></p>

__Auteurs__ : Chloé Gobé, Xavier Rettel, Mounia Slassi  <br>
__Github du projet__ : `https://github.com/ChloeGobe/art_deep_ecp`

## Index
1. [Description](#description)
2. [Articles utilisés](#docs)
3. [Requirements techniques](#requirements)
4. [Contenu du projet](#arborescence)
5. [Installation et lancement](#installation)


## <a name="description"></a>1. Description
Projet de Deep Learning pour la reconnaissance de styles de peintures

## <a name="docs"></a>2. Articles utilisés
- Wei Ren Tan, Chee Seng Chan, Hernàn E Aguirre, and Kiyoshi Tanaka. "*Ceci n’est pas une pipe: A deep convolutional network for fine-art paintings classification*." In Image Processing (ICIP), 2016 IEEE International Conference on, pages 3703–3707. IEEE, 2016. :  [article](http://ieeexplore.ieee.org/iel7/7527113/7532277/07533051.pdf)  

- Adrian Lecoutre, Benjamin Negrevergne, Florian Yger "*Recognizing Art Style Automatically in painting with deep learning*", JMLR: Workshop and Conference Proceedings 80:1–17, 2017 [article](http://www.lamsade.dauphine.fr/~bnegrevergne/webpage/documents/2017_rasta.pdf)  

## <a name="requirements"></a>3. Requirements techniques
- Keras 2.1.4
- Tensorflow 1.6
- Pillow
- scikit-learn

## 4. <a name="arborescence"></a>Structure du projet

- **Articles et documents** quelques PDFs qui nous ont servi.
- **Ceci n'est pas une pipe** clone du répertoire éponyme.
- **RASTA** clone du répertoire éponyme.
- **data** : la version très réduite du dataset, pour se faire une idée des données manipulées. Pour faire tourner le code ci-dessous, il faut disposer du dataset complet.


## <a name="installation"></a>5. Installation et lancement
- Pour télécharger le dataset (20Go): 

      cd data
      wget www.lamsade.dauphine.fr/~bnegrevergne/webpage/software/rasta/wikipaintaings_full.tgz
      tar xzvf wikipaintings_full.tgz
      cd ../
- Pour lancer un entraînement :

      python -u transfer_resnet.py
- Pour lancer l'évaluation sur le set de test :

      python -u evaluation.py --data_path ./data/wikipaintings_test --model_path ./model.h5
- La commande précédente affiche l'accuracy Top-1, Top-3 et Top-5 et crée les fichiers ``y_pred.npy`` et ``y_true.npy`` qui correspondent aux prédictions et à la ground truth.
- Pour générer la matrice de confusion :

      python confusion.py
