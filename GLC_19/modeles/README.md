# Modèles implémentés pour GLC2019: 

## Structure et organisation du code. 
Pour run et entraîné les modèles il faut avoir accès au fichier CSV, ainsi qu'au dossier
où sont stocké les tenseurs environnementaux.
Dans chaque fichier on a une fonction run où l'on met les paramètres du modèles, et sur quels fichiers csv on veut entraîner notre modèle.
On a également un paramètre test_size puisque l'on dispose pas encore de fichier test donc on est obligé de partitionner notre dataset.

## Classifier

*classifier.py*
Tout les modèles héritent de cette classe Classifier, on a donné la structure de la classe ainsi que les fonctions
pour calculer les scores MRR et Top30. 

### Score MRR
#### Mean Reciprocal Rank
Le MRR est une mesure statistique pour évaluer des modèles qui renvoient une liste ordonnée par probabilité de labels (ranking). Le rang réciproque d'une réponse à une requête est l'inverse multiplicatif du rang de la première réponse correcte. On a 1 pour la première place, 1/2 pour la deuxième, 1/3 pour la troisième etc...
Le rang réciproque moyen est la moyenne des rangs réciproques des résultats pour un échantillon de requêtes Q.

### Top30
Beaucoup plus souple que le MRR. 
Avec nos modèles pour chaque entrée du test on renvoie une liste de 30 espèces candidates. 
Si l'espèce à prédire est dans cette liste alors 1, sinon 0. On fait la moyenne et ça nous donne notre score.

C'est cette métrique qui va être utilisé pour l'évaluation des modèles dans le projet. 
Mais on utilise la MRR pour comparer avec les résultats des années précédentes puisqu'ils utilisaient cette mesure. 

## Modèle random.

*randommodel.py*
Simple modèle qui renvoie une liste d'espèces de taille *ranking_size* en tirant au hasard parmis les espèces qu'il a vu 
dans le train_set.


## Modèle Fréquentielle. 

*frequence.py*
Simple modèle qui compte les occurrences les plus présentes dans le training_set et renvoie naïvement pour chaque entrée du test 
la même liste de taille *ranking_size* ordonnée par plus grande fréquence.


## Modèle vectoriel.

*vector_model.py*

Modèle vectoriel basé sur les plus proches voisins environnementaux.
On vectorise nos tenseurs pour ne plus avoir un descriptif d'environnement de taille 64x64x33 mais uniquement de taille 33.
La vectorisation se fait en faisant une moyenne des pixels centraux pour les 33 images. Le paramètre *window_size* permet de déterminer
le nombre de pixels que l'on prend en compte pour la moyenne. 
Ensuite une fois que l'on a nos vecteurs du **training_set**, on vectorise ceux de notre **test_set** et on cherche la distance euclidienne minimum 
avec les vecteurs du **training_set**. On renvoie les *ranking_size* plus proches voisins. 

### Problème dans notre modèle. 
•Il faut qu'on diminue la taille de la matrice des distances calculer dans la méthode *predict* avec la fonction *scipy.spatial.distance.cdist(test_vectors,self.train_vectors , metric='euclidean')*
• Ce modèle est très lent à entraîner.

## Modèle K - means.

*kmeans.py*

Comme pour le modèle vectoriel on ne travaille pas sur les tenseurs 64x64x33 mais sur des vecteurs de taille 33. Avec toujours le paramètre *window_size* pour déterminer le nombre de pixels à prendre en compte dans la moyenne. (méthode : *tensors_to_vectors*)
K - means est un algo de clustering non supervisé. Comme l'on travail avec des training set labelisé on a du transformer notre espace pour qu'il soit délabélisé afin d'appliquer K - means, avec la bibliothèque **sklearn**.
Nos paramètres sont donc la *window_size* et *nb_cluster* qui va déterminer le nombre de cluster que l'on a pour le k-means. 
Pour construire notre espace non labélisé et aplliquer K - means : 
- On a decidé de construire un vecteur environnemental caractéristique pour chaque espèce. 
- On a compté le nombre d'espèces différente dans le dataset. 
- On fait la moyenne de tout les vecteurs environnementaux d'une espèce qui sera notre vecteur caractéristique. On stock dans une variable *species_matrix* 
- On applique notre K-means en gardant en mémoire les espèces qui appartiennent à un même cluster. 
- On prédit sur nos entrées de test à quel cluster ils appartiennent. 

Ensuite deux méthodes pour sélectionner les espèces du bon cluster. Hard et Soft. 

### Hard K - means. 
Quand K-means nous renvoie le cluster auquel notre entrée test appartient on regarde quels autres espèces du training_set appartiennent à ce cluster. 
Si le nombre d'espèces de ce cluster est plus grand que *ranking_size* on prend les espèces qui sont les plus fréquentes dans le training_set. 

### Soft K - means. 
En cours. 

#### Question : 
Comment assigner les probabilités pour les espèces que l'on veut classer ? 











