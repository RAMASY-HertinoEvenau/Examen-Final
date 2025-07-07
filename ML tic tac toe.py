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

if __name__ == "__main__":
    print("Génération du jeu de données...")
    etats_plateau, mouvements_optimaux = generer_jeu_donnees(1000)
    print(f"Dataset généré : {len(etats_plateau)} états de plateau")