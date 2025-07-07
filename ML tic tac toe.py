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

    def entrainer(self, X, y, taux_apprentissage=0.01, epochs=1000):
        for _ in range(epochs):
            self.propagation_avant(X)
            y_one_hot = np.zeros((len(y), 9))
            y_one_hot[np.arange(len(y)), y] = 1
            erreur_sortie = self.sortie - y_one_hot
            grad_sortie = erreur_sortie
            grad_poids_cachee_sortie = np.dot(self.cachee.T, grad_sortie)
            grad_biais_sortie = np.sum(grad_sortie, axis=0, keepdims=True)
            grad_cachee = np.dot(grad_sortie, self.poids_cachee_sortie.T) * self.cachee * (1 - self.cachee)
            grad_poids_entree_cachee = np.dot(X.T, grad_cachee)
            grad_biais_cache = np.sum(grad_cachee, axis=0, keepdims=True)
            self.poids_entree_cachee -= taux_apprentissage * grad_poids_entree_cachee
            self.biais_cache -= taux_apprentissage * grad_biais_cache
            self.poids_cachee_sortie -= taux_apprentissage * grad_poids_cachee_sortie
            self.biais_sortie -= taux_apprentissage * grad_biais_sortie

    def predire(self, X):
        return np.argmax(self.propagation_avant(X), axis=1)

def afficher_plateau(plateau):
    symboles = {0: ' ', 1: 'X', -1: 'O'}
    for i in range(0, 9, 3):
        print(f"{symboles[plateau[i]]} | {symboles[plateau[i+1]]} | {symboles[plateau[i+2]]}")
        if i < 6:
            print("-" * 9)

def jouer_partie(modele):
    plateau = initialiser_plateau()
    print("Bienvenue au Tic-Tac-Toe ! Vous êtes O, l'IA est X.")
    print("Entrez un numéro de 0 à 8 pour jouer.")
    afficher_plateau(np.array([i for i in range(9)]))
    
    while True:
        # Tour de l'IA (X)
        mouvements_disponibles = [i for i in range(9) if plateau[i] == 0]
        if mouvements_disponibles:
            etat_plateau = plateau.reshape(1, -1)
            mouvement = modele.predire(etat_plateau)[0]
            if mouvement in mouvements_disponibles:
                plateau[mouvement] = 1
                print("\nTour de l'IA :")
                afficher_plateau(plateau)
                if verifier_gagnant(plateau, 1):
                    print("L'IA (X) gagne !")
                    break
                if 0 not in plateau:
                    print("Match nul !")
                    break
        
        # Tour du joueur (O)
        while True:
            try:
                mouvement = int(input("Votre tour (0-8) : "))
                if mouvement in mouvements_disponibles:
                    plateau[mouvement] = -1
                    print("\nVotre tour :")
                    afficher_plateau(plateau)
                    if verifier_gagnant(plateau, -1):
                        print("Vous (O) gagnez !")
                        break
                    if 0 not in plateau:
                        print("Match nul !")
                        break
                    break
                else:
                    print("Mouvement invalide. Essayez encore.")
            except ValueError:
                print("Entrez un numéro valide (0-8).")
        if verifier_gagnant(plateau, -1):
            break

if __name__ == "__main__":
    print("Génération du jeu de données...")
    etats_plateau, mouvements_optimaux = generer_jeu_donnees(1000)
    print("Entraînement du modèle...")
    modele = ReseauNeurones()
    modele.entrainer(etats_plateau, mouvements_optimaux, taux_apprentissage=0.01, epochs=1000)
    print("Modèle entraîné. Prêt à jouer !")
    jouer_partie(modele)
    