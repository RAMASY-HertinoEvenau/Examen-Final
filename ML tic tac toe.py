import numpy as np
import random

# Fonction pour initialiser un plateau vide
def initialiser_plateau():
    return np.zeros(9, dtype=int)

# Fonction pour vérifier le gagnant
def verifier_gagnant(plateau, joueur):
    combinaisons_gagnantes = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Lignes
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Colonnes
        [0, 4, 8], [2, 4, 6]              # Diagonales
    ]
    for combinaison in combinaisons_gagnantes:
        if all(plateau[i] == joueur for i in combinaison):
            return True
    return False

# Fonction Minimax pour déterminer le meilleur mouvement
def minimax(plateau, profondeur, est_maximisant):
    if verifier_gagnant(plateau, 1):
        return 1
    if verifier_gagnant(plateau, -1):
        return -1
    if 0 not in plateau:
        return 0

    if est_maximisant:
        meilleur_score = -float('inf')
        for i in range(9):
            if plateau[i] == 0:
                plateau[i] = 1
                score = minimax(plateau, profondeur + 1, False)
                plateau[i] = 0
                meilleur_score = max(score, meilleur_score)
        return meilleur_score
    else:
        meilleur_score = float('inf')
        for i in range(9):
            if plateau[i] == 0:
                plateau[i] = -1
                score = minimax(plateau, profondeur + 1, True)
                plateau[i] = 0
                meilleur_score = min(score, meilleur_score)
        return meilleur_score

# Générer le dataset avec Minimax
def generer_jeu_donnees(nb_parties):
    etats_plateau, mouvements_optimaux = [], []
    for _ in range(nb_parties):
        plateau = initialiser_plateau()
        while 0 in plateau and not verifier_gagnant(plateau, 1) and not verifier_gagnant(plateau, -1):
            mouvements_disponibles = [i for i in range(9) if plateau[i] == 0]
            if not mouvements_disponibles:
                break
            meilleur_mouvement = None
            meilleur_score = -float('inf')
            for mouvement in mouvements_disponibles:
                plateau[mouvement] = 1
                score = minimax(plateau, 0, False)
                plateau[mouvement] = 0
                if score > meilleur_score:
                    meilleur_score = score
                    meilleur_mouvement = mouvement
            etats_plateau.append(plateau.copy())
            mouvements_optimaux.append(meilleur_mouvement)
            plateau[meilleur_mouvement] = 1
            if 0 in plateau and not verifier_gagnant(plateau, 1):
                plateau[random.choice(mouvements_disponibles)] = -1
    return np.array(etats_plateau), np.array(mouvements_optimaux)

class ReseauNeurones:
    def __init__(self, taille_entree=9, taille_cachee=32, taille_sortie=9):
        self.poids_entree_cachee = np.random.randn(taille_entree, taille_cachee) * 0.1
        self.biais_cache = np.zeros((1, taille_cachee))
        self.poids_cachee_sortie = np.random.randn(taille_cachee, taille_sortie) * 0.1
        self.biais_sortie = np.zeros((1, taille_sortie))

    def sigmoide(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def propagation_avant(self, X):
        self.cachee = self.sigmoide(np.dot(X, self.poids_entree_cachee) + self.biais_cache)
        self.sortie = self.softmax(np.dot(self.cachee, self.poids_cachee_sortie) + self.biais_sortie)
        return self.sortie

if __name__ == "__main__":
    print("Génération du jeu de données...")
    etats_plateau, mouvements_optimaux = generer_jeu_donnees(1000)
    print("Initialisation du réseau de neurones...")
    modele = ReseauNeurones()
    sorties = modele.propagation_avant(etats_plateau[:5])
    print(f"Sorties pour 5 premiers états : \n{sorties}")