import numpy as np

def quota_ricorrenti(x):
    # coppia non orientata: A-B e B-A sono la stessa coppia
    u = np.where(x["mittente"] < x["destinatario"], x["mittente"], x["destinatario"])
    v = np.where(x["mittente"] < x["destinatario"], x["destinatario"], x["mittente"])
    n_thread = x.assign(u=u, v=v).groupby(["finestra", "u", "v"])["thread"].nunique()
    return (n_thread >= 2).groupby(level="finestra").mean()

def metriche(c):          # c = matrice (finestre x utenti) con il numero di commenti
    att = c > 0
    pers = att[:-1] & att[1:]                       # attivo in k e in k+1
    ricambio = 1 - pers.sum(1) / att[1:].sum(1)
    quota = (c[1:] * pers).sum(1) / c[1:].sum(1)
    return ricambio, quota

