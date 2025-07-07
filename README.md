# Examen-Final

Répartition des tâches

Le projet a été développé par une équipe de cinq personnes, chacune ayant contribué à une partie spécifique du code :

RAMASY Hertino Evenau : Génération du dataset
Implémentation de la fonction generer_jeu_donnees et les fonctions associées (initialiser_plateau, verifier_gagnant, minimax). Ces fonctions génèrent un dataset d'états de plateau et de mouvements optimaux en simulant des parties avec l'algorithme Minimax.

TSISANDAINA Finaritra Silvio Michael : Réseau de neurones (structure)
Implémentation de la classe ReseauNeurones, y compris l'initialisation des poids, les fonctions d'activation (sigmoide et softmax), et la propagation avant (propagation_avant).

TSIFARIA Manados Dolsain : Entraînement du réseau
Développement de la méthode d'entraînement (entrainer) de la classe ReseauNeurones, implémentant la rétropropagation et la mise à jour des poids avec la descente de gradient.

RAZAFINDRAINIBE Santatra Mirado : Prédiction et intégration
Implémentation de la méthode de prédiction (predire) du réseau de neurones et intégré le modèle dans la fonction jouer_partie pour permettre à l'IA de choisir des mouvements pendant le jeu.

OMAR ABDOUL ANZIZ TAJ Tsiory : Interface utilisateur
Création d'un interface textuelle, incluant la fonction afficher_plateau et la logique d'interaction avec le joueur humain dans jouer_partie, ainsi que les messages d'affichage pour guider l'utilisateur.