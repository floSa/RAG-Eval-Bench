"""Annotation manuelle de la FIDELITE, le seul poste que la calibration
automatique ne couvre pas.

`ragbench calibrate` compare les juges a `native.answer/contains`, derive des
reponses gold. Cette reference porte sur la JUSTESSE. Elle ne dit rien de la
fidelite : une reponse peut etre parfaitement fidele aux passages remontes et
fausse, parce que le mauvais passage a ete remonte. Valider un juge de fidelite
demandait donc une annotation a la main — c'est ce que fige ce fichier.

Protocole : 13 reponses NON abstenues du run 24, jugees avec les passages
remontes INTEGRAUX sous les yeux (pas de troncature). Critere : chaque
affirmation de la reponse est-elle soutenue par les passages ? 1 = oui.
Ce n'est PAS « la reponse est correcte ».

Resultat : 11/13 fideles (0.846). Les deux echecs et, surtout, les trois
reponses FAUSSES MAIS FIDELES sont documentes ci-dessous — ce sont eux qui
prouvent empiriquement, sur ce corpus, que fidelite != justesse.

    uv run python scripts/annotation_fidelite.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ragbench import db  # noqa: E402

RUN_ID = 24
ANNOTATOR = "claude-opus-5"
METRIC = "faithful"

# 1 = toutes les affirmations de la reponse sont soutenues par les passages.
VERDICTS: dict[int, int] = {
    87: 1,   # « $26.3 md en 2021 dans les deux articles » : verifie mot pour mot
    105: 1,  # les deux passages attestent bien une position de tete
    127: 1,  # Eras Tour, Arrowhead, paparazzi, Kelce : tout est dans les passages
    135: 1,  # « not consistently candid », theorie dominante : verbatim TechCrunch
    149: 1,  # « No » coherent avec les passages (economie qui ralentit, pas l'inverse)
    157: 1,  # « No » coherent : aucun passage Sporting News ne parle du sack
    192: 1,  # Ellison comme paravent, Alameda/FTX, fraude : tout est atteste
    218: 1,  # y compris le detail des messages detruits, repris verbatim
    465: 0,  # ECHEC : repond « Google » a une question dont les criteres
             # designent Apple (passage #4, The Verge). Le modele a retenu
             # l'entite la plus frequente du contexte, pas celle qui satisfait
             # les criteres. Pas une hallucination — une attribution non fondee.
    600: 1,  # FAUSSE MAIS FIDELE : le passage Sporting News remonte ne parle
             # pas de Cease/Burnes/Glasnow. Repondre « No » est fonde sur ce
             # qui a ete remonte. L'echec est au retrieval, pas au generateur.
    655: 0,  # ECHEC : les passages disent explicitement que le Guardian precise
             # le statut d'approbation, et que Business Line cite un second
             # medicament approuve. La reponse « Yes » les contredit.
    720: 1,  # FAUSSE MAIS FIDELE : le passage Raiders remonte montre une prise
             # d'avance, pas une egalisation. « No » est fonde sur l'extrait.
    864: 1,  # FAUSSE MAIS FIDELE : aucun passage Sporting News sur le defi des
             # 2 000 yards n'a ete remonte.
}

NOTE = ("Annotation manuelle Claude Opus 5, passages remontes integraux sous les yeux. "
        "Critere : chaque affirmation de la reponse est-elle soutenue par les passages ?")

SQL = """
SELECT p.question_id, q.answer AS gold, p.answer AS predicted
FROM predictions p
JOIN questions q ON q.id = p.question_id
WHERE p.run_id = %s AND p.question_id = ANY(%s)
ORDER BY p.question_id
"""


def main() -> int:
    qids = sorted(VERDICTS)
    with db.connect() as conn:
        rows = conn.execute(SQL, (RUN_ID, qids)).fetchall()
        if len(rows) != len(qids):
            print(f"ATTENTION : {len(rows)} predictions pour {len(qids)} verdicts figes.")

        for r in rows:
            conn.execute(
                "INSERT INTO annotations (run_id, question_id, metric, value, annotator, notes) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (run_id, question_id, metric, annotator) "
                "DO UPDATE SET value = EXCLUDED.value",
                (RUN_ID, r["question_id"], METRIC, float(VERDICTS[r["question_id"]]),
                 ANNOTATOR, NOTE),
            )
        conn.commit()

    n = len(VERDICTS)
    fideles = sum(VERDICTS.values())
    print(f"n = {n}   ({ANNOTATOR}, run {RUN_ID}, metrique '{METRIC}')")
    print(f"Fidelite mesuree : {fideles / n:.3f}  ({fideles}/{n})")
    print()
    print("Non fideles :")
    for qid in (465, 655):
        print(f"  qid={qid}")
    print()
    print("Fausses MAIS fideles — l'echec est au retrieval, pas au generateur :")
    for qid in (600, 720, 864):
        print(f"  qid={qid}")
    print()
    print(f"{n} annotations enregistrees.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
