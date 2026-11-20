import numpy as np
import pandas as pd

def quota_ricorrenti(x):
    # coppia non orientata: A-B e B-A sono la stessa coppia
    u = np.where(x["mittente"] < x["destinatario"], x["mittente"], x["destinatario"])
    v = np.where(x["mittente"] < x["destinatario"], x["destinatario"], x["mittente"])
    n_thread = x.assign(u=u, v=v).groupby(["finestra", "u", "v"])["thread"].nunique()
    return (n_thread >= 2).groupby(level="finestra").mean()

def metriche(c, min_events=1):   # c = matrice (finestre x utenti) con il numero di commenti
    att = c >= min_events
    A, B = att[:-1], att[1:]
    comuni = (A & B).sum(1)

    R = 1 - comuni / (A | B).sum(1)
    f_out = 1 - comuni / A.sum(1)
    f_in = 1 - comuni / B.sum(1)
    quota_A = (c[:-1] * (A & B)).sum(1) / c[:-1].sum(1)
    quota_B = (c[1:] * (A & B)).sum(1) / c[1:].sum(1)
    return R, f_out, f_in, quota_A, quota_B
    
def ricambio(data_inizio, giorni, n_finestre, interazioni, min_events=1):

    t0 = pd.Timestamp(data_inizio, tz="UTC")
    step = pd.Timedelta(days=giorni)

    # numero della finestra di ogni riga; scarta mittenti mancanti e righe fuori dalle finestre
    ev = interazioni.dropna(subset=["mittente"]).copy()
    ev["w"] = (ev["t"] - t0) // step
    ev = ev[(ev["w"] >= 0) & (ev["w"] < n_finestre)]
    ev["w"] = ev["w"].astype(int)

    # numero di risposte per finestra, utente e tipo di destinatario
    per_tipo = (ev.groupby(["w", "mittente", "tipo_dest"]).size()
                  .unstack(fill_value=0)
                  .reindex(columns=["post", "commento"], fill_value=0))

    # totali per finestra
    totali = per_tipo.groupby(level="w").sum().reindex(range(n_finestre), fill_value=0)

    # utenti attivi per finestra (almeno min_events risposte)
    tot_utente = per_tipo.sum(axis=1)
    attivi = {k: set(c[c >= min_events].droplevel("w").index)
              for k, c in tot_utente.groupby(level="w")}

    def quota(k, utenti):
        sub = per_tipo.loc[k]
        return sub[sub.index.isin(utenti)].sum() / sub.sum()

    righe = []
    for k in range(n_finestre - 1):
        A, B = attivi.get(k, set()), attivi.get(k + 1, set())
        if not A or not B:
            continue
        comuni = A & B
        qA, qB = quota(k, comuni), quota(k + 1, comuni)
        righe.append({
            "inizio_A": t0 + k * step,
            "inizio_B": t0 + (k + 1) * step,
            "n_A": len(A),
            "n_B": len(B),
            "n_comuni": len(comuni),
            "risp_post_A": totali.loc[k, "post"],
            "risp_comm_A": totali.loc[k, "commento"],
            "risp_post_B": totali.loc[k + 1, "post"],
            "risp_comm_B": totali.loc[k + 1, "commento"],
            "R": 1 - len(comuni) / len(A | B),
            "f_out": len(A - B) / len(A),
            "f_in": len(B - A) / len(B),
            "quota_utenti_A": len(comuni) / len(A),
            "quota_risp_post_A": qA["post"],
            "quota_risp_comm_A": qA["commento"],
            "quota_utenti_B": len(comuni) / len(B),
            "quota_risp_post_B": qB["post"],
            "quota_risp_comm_B": qB["commento"],
        })
    return pd.DataFrame(righe)

def gini_risposte(interazioni, community, inizio, fine):
    """Gini delle risposte ricevute dagli utenti di una community in una finestra [inizio, fine).
    La popolazione include chi ha scritto nella finestra e chi ha ricevuto risposte, anche chi ne ha 0."""

    inizio, fine = pd.Timestamp(inizio, tz="UTC"), pd.Timestamp(fine, tz="UTC")

    # righe della community nella finestra, con mittente e destinatario noti
    ev = interazioni[(interazioni["community"] == community)
                     & (interazioni["t"] >= inizio) & (interazioni["t"] < fine)]
    ev = ev.dropna(subset=["mittente", "destinatario"])

    # risposte ricevute (niente auto-risposte)
    valide = ev[ev["mittente"] != ev["destinatario"]]
    ricevute = valide["destinatario"].value_counts()

    # popolazione di riferimento: chi scrive e chi riceve, anche con 0 risposte
    popolazione = pd.concat([ev["mittente"], ev["destinatario"]]).unique()

    x = np.sort(ricevute.reindex(popolazione, fill_value=0).to_numpy(float))
    n, tot = len(x), x.sum()
    gini = ((2 * np.arange(1, n + 1) - n - 1) * x).sum() / (n * tot) if tot > 0 else np.nan

    return {"community": community, "inizio": inizio, "n_utenti": n, "n_risposte": int(tot),
            "quota_senza_risposte": float((x == 0).mean()) if n else np.nan, "gini": gini}

def coppie_per_finestra(interazioni, inizio, fine, freq="7D", orientate=False):
    """Per ogni finestra: quante coppie di utenti interagiscono in 2+ thread diversi.
    Interazione = A risponde a B (al suo commento, o al suo post se è una risposta diretta).
    orientate=False: A->B e B->A sono la stessa coppia."""

    inizio, fine = pd.Timestamp(inizio, tz="UTC"), pd.Timestamp(fine, tz="UTC")
    freq = pd.Timedelta(freq)

    # righe nella finestra, con mittente e destinatario noti (niente auto-risposte)
    ev = interazioni.dropna(subset=["mittente", "destinatario"])
    ev = ev[(ev["mittente"] != ev["destinatario"])
            & (ev["t"] >= inizio) & (ev["t"] < fine)].copy()

    # coppia (u, v): se non orientata, ordine alfabetico così A-B coincide con B-A
    a, b = ev["mittente"].astype(str), ev["destinatario"].astype(str)
    if orientate:
        ev["u"], ev["v"] = a, b
    else:
        ev["u"], ev["v"] = np.where(a < b, a, b), np.where(a < b, b, a)

    ev["finestra"] = (ev["t"] - inizio) // freq

    # in quanti thread diversi interagisce ogni coppia, per finestra
    thread_per_coppia = ev.groupby(["finestra", "u", "v"])["thread"].nunique()

    n_finestre = int(np.ceil((fine - inizio) / freq))
    out = pd.DataFrame({
        "n_interazioni": ev.groupby("finestra").size(),
        "n_coppie": thread_per_coppia.groupby(level="finestra").size(),
        "coppie_2plus_thread": (thread_per_coppia >= 2).groupby(level="finestra").sum(),
    }).reindex(range(n_finestre), fill_value=0)

    out["quota_coppie_ricorrenti"] = (out["coppie_2plus_thread"] / out["n_coppie"]).where(out["n_coppie"] > 0)
    out["data"] = [inizio + i * freq for i in out.index]
    return out.reset_index(drop=True)[["data", "n_interazioni", "n_coppie",
                                       "coppie_2plus_thread", "quota_coppie_ricorrenti"]]