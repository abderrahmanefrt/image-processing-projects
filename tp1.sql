CREATE TABLE Agriculteur (
    agriculteur_id NUMBER PRIMARY KEY,
    nom VARCHAR2(100) NOT NULL,
    localisation VARCHAR2(100),
    taille_exploitation NUMBER,
    culture_principale VARCHAR2(100)
);

CREATE TABLE Produit_Alimentaire (
    produit_id NUMBER PRIMARY KEY,
    nom VARCHAR2(100) NOT NULL,
    categorie VARCHAR2(50),
    valeur_nutritionnelle VARCHAR2(50)
);

CREATE TABLE Production (
    production_id NUMBER PRIMARY KEY,
    agriculteur_id NUMBER NOT NULL,
    produit_id NUMBER NOT NULL,
    quantite_produite NUMBER DEFAULT 0,
    saison VARCHAR2(50),
    CONSTRAINT fk_prod_agriculteur FOREIGN KEY (agriculteur_id) REFERENCES Agriculteur(agriculteur_id),
    CONSTRAINT fk_prod_produit FOREIGN KEY (produit_id) REFERENCES Produit_Alimentaire(produit_id)
);

CREATE TABLE Marche (
    marche_id NUMBER PRIMARY KEY,
    nom VARCHAR2(100) NOT NULL,
    localisation VARCHAR2(100),
    type_marche VARCHAR2(20)
);

CREATE TABLE Approvisionnement (
    agriculteur_id NUMBER NOT NULL,
    produit_id NUMBER NOT NULL,
    marche_id NUMBER NOT NULL,
    date_approvisionnement DATE,
    quantite_fournie NUMBER,
    CONSTRAINT pk_approvisionnement PRIMARY KEY (agriculteur_id, produit_id, marche_id, date_approvisionnement),
    CONSTRAINT fk_appro_agriculteur FOREIGN KEY (agriculteur_id) REFERENCES Agriculteur(agriculteur_id),
    CONSTRAINT fk_appro_produit FOREIGN KEY (produit_id) REFERENCES Produit_Alimentaire(produit_id),
    CONSTRAINT fk_appro_marche FOREIGN KEY (marche_id) REFERENCES Marche(marche_id)
);