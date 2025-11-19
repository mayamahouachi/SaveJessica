# Explication stratégie LinUCBStrategy

## I. Phase exploration du comportement des planètes : 
L’exploration a montré que les planètes ont un comportement qui se rapproche d'une sinusoidale bruitée :

- Planet 2 suit un courbe sinusoïdal avec un peu de bruit mais une période nette = 200.

- Planet 1 présente un cycle plus instable de période = 20 avec des variations irrégulières.

- Planet 0 est largement bruitée avec de petites oscillations locales de période =10


## II. phase implémentation de la stratégie LinUCBStrategy:

La stratégie modélise chaque planète comme un bandit manchot linéaire dont le comportement est périodique en utilisant les périodes estimées lors de la phase d’exploration.
L’objectif est de capturer ces cycles pour prédire plus précisément les chances de survie et envoyer le bon nombre de Morties.
La stratégie repose sur : 

**1. Encodage des features:**

On  modélise le comportement de chaque planète par la construction d'un vecteur de features x basé sur des harmoniques sinusoïdales qui permettent d’encoder  où l’on se trouve dans le cycle périodique de la planète, tout en capturant des variations plus fines à l’intérieur du cycle.

**2. Application de LinUCB par planète**

Chaque planète possède :

- une matrice A qui accumule les produits de features 

- un vecteur b qui accumule les récompenses

LinUCB reconstruit alors une probabilité de survie prédite et un bonus d’exploration.

**3. Heuristique locale :**

Pour maximiser le gain, on apprend pendant l’épisode des statistiques par phase.
Si une phase se révèle dangereuse, le score UCB est pénalisé pour éviter les steps à haut risque.
Cela permet d’adapter le nombre de Morties envoyés en fonction de la proba prédite, du nombre de visites sur la planète, de la qualité locale de la phase et du stade de l’épisode.


## III. Résultat

Le meilleur gain obtenu avec cette stratégie est 88.7% 
avec
**Login : MayaMahouachi**


## IV . Exécution du script : 
- python main.py : exécute la stratégie et l’analyse post-exécution.
- python main.py --explore : relance la phase d’exploration pure + son analyse pré-stratégie puis exécute de  la stratégie et et visualiser ses plots.
- python main.py --no-run : saute l'exécution de la stratégie

- python main.py --no-post : saute l'analyse post-exécution.

