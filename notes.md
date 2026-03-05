Idées à implémenter

Vis à vis de la qualité :
- Pondération de l'observation en fonction de la qualité si jamais l'image est floue ou non
- Si ça ne marche pas, explication en fonction de la qualité
111 c'est bien
000 c'est degueulasse
- Kfold au lieu de train val test split (un stratified K-Fold serait trop dur)
- Cartes de poids (par rapport à la ground truth (ex: attention accrue sur les zones fines), Charlotte est l'expert)
- Faire le lien entre segmentation et maladie (trois types de maladies identifiables dans le nom de l'image)
    - Modifier le VesselDataset pour récupérer le nom de la maladie (DONE)
    - Donner la segmentation des vaisseaux sanguins à un MLP et faire la classification
- Tester SAM et la version Med-SAM


A faire :
- voir si on fait une fonction qui permet de faire des logs automatiques des expériences (sous forme de config.txt)
- vérifier si les labels sont bien binarisés (possible modification sur le calcul des métriques dans predict)
- faire un resize explicite avant le calcul des métriques (je ne sais pas si le calcul actuel dans predict() est correct)
- ATTENTION au resize des labels qui les rendent continus