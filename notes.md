Idées à implémenter

Vis à vis de la qualité :
- Pondération de l'observation en fonction de la qualité si jamais l'image est floue ou non
- Si ça ne marche pas, explication en fonction de la qualité
111 c'est bien
000 c'est degueulasse
- Kfold au lieu de train val test split (un stratified K-Fold serait trop dur)
- Early stopping :
    - pas de chgmt depuis 10 epochs
    - hausse de la val losses depuis 5 epochs
- Cartes de poids (par rapport à la ground truth (ex: attention accrue sur les zones fines), Charlotte est l'expert)
- Faire le lien entre segmentation et maladie (trois types de maladies identifiables dans le nom de l'image)
    - Modifier le VesselDataset pour récupérer le nom de la maladie
- Tester SAM et la version Med-SAM
