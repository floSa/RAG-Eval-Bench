"""Etalonnage du plafond : que donnerait un tres grand juge ?

Les quatre juges locaux testes plafonnent sous le seuil d'utilisabilite
(kappa > 0,6). Restait a savoir si c'est une limite de la TAILLE des modeles
ou de la TACHE — certaines questions du corpus sont peut-etre indecidables.

Ce script fige les 53 verdicts rendus a la main par Claude Opus 5 sur le
run 24, en aveugle (la colonne de reference n'a pas ete consultee avant de
trancher), et recalcule le kappa contre la verite terrain.

Resultat : kappa = +0.958, contre +0.458 pour le meilleur juge local. La
tache est donc jugeable ; ce sont les petits modeles qui ne la jugent pas.

Le seul desaccord sur 53 incrimine la REFERENCE, pas le juge : `contains`
compte faux une reponse correcte donnee sous forme abregee (`Bankman-Fried`
pour un gold `Sam Bankman-Fried`). Toutes les valeurs de `contains` publiees
par le banc sont donc des bornes basses.

Ce juge n'est PAS deployable : appel sortant, il viole la contrainte
on-premise. C'est un etalon mesure une fois sur un corpus public.

    uv run python scripts/calibration_juge_externe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ragbench import db  # noqa: E402
from ragbench.stats import cohen_kappa  # noqa: E402

RUN_ID = 24
ANNOTATOR = "claude-opus-5"

# Critere identique a `native.answer/contains` : la reponse restitue-t-elle la
# reponse gold ? 1 = oui, 0 = non (fausse OU abstention).
VERDICTS = {
    18: 0, 22: 0, 23: 0, 35: 0, 68: 0, 74: 0, 87: 1, 88: 0, 89: 0, 95: 0,
    100: 0, 105: 1, 127: 1, 135: 1, 149: 1, 157: 1, 192: 1, 218: 1, 241: 0,
    259: 1, 272: 0, 278: 1, 287: 0, 317: 0, 325: 0, 330: 1, 332: 1, 362: 1,
    383: 0, 402: 0, 423: 0, 429: 1, 463: 0, 465: 0, 516: 1, 530: 0, 540: 0,
    554: 1, 556: 0, 564: 0, 587: 0, 600: 0, 628: 0, 652: 0, 655: 0, 656: 0,
    658: 0, 689: 1, 694: 1, 699: 0, 705: 0, 711: 1, 720: 0,
}

REFERENCE_SQL = """
SELECT p.question_id, q.answer AS gold, p.answer AS predicted, s.value AS ref
FROM predictions p
JOIN questions q ON q.id = p.question_id
JOIN scores s ON s.run_id = p.run_id AND s.question_id = p.question_id
             AND s.evaluator = 'native.answer' AND s.metric = 'contains'
WHERE p.run_id = %s
ORDER BY p.question_id
"""

NOTE = ("Jugement manuel Claude Opus 5, en aveugle (reference non consultee "
        "avant verdict). Critere identique a native.answer/contains.")


def main() -> int:
    with db.connect() as conn:
        rows = [r for r in conn.execute(REFERENCE_SQL, (RUN_ID,)).fetchall()
                if r["question_id"] in VERDICTS]

        if len(rows) != len(VERDICTS):
            print(f"ATTENTION : {len(rows)} predictions retrouvees pour "
                  f"{len(VERDICTS)} verdicts figes. Le run {RUN_ID} a-t-il change ?")

        mine = [VERDICTS[r["question_id"]] for r in rows]
        ref = [int(round(r["ref"])) for r in rows]
        n = len(rows)

        for r, v in zip(rows, mine):
            conn.execute(
                "INSERT INTO annotations (run_id, question_id, metric, value, annotator, notes) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (run_id, question_id, metric, annotator) "
                "DO UPDATE SET value = EXCLUDED.value",
                (RUN_ID, r["question_id"], "correct", float(v), ANNOTATOR, NOTE),
            )
        conn.commit()

    accord = sum(a == b for a, b in zip(mine, ref)) / n
    genereux = sum(a == 1 and b == 0 for a, b in zip(mine, ref)) / n
    severe = sum(a == 0 and b == 1 for a, b in zip(mine, ref)) / n

    print(f"n = {n}   ({ANNOTATOR}, run {RUN_ID})")
    print(f"Justesse selon le juge   : {sum(mine) / n:.3f}  ({sum(mine)}/{n})")
    print(f"Justesse selon `contains`: {sum(ref) / n:.3f}  ({sum(ref)}/{n})")
    print()
    print(f"kappa de Cohen : {cohen_kappa(mine, ref):+.3f}")
    print(f"Accord brut    : {accord:.3f}")
    print(f"Trop genereux  : {genereux:.3f}")
    print(f"Trop severe    : {severe:.3f}")

    desaccords = [(r, a, b) for r, a, b in zip(rows, mine, ref) if a != b]
    print(f"\n--- {len(desaccords)} desaccord(s) ---")
    for r, a, b in desaccords:
        print(f"qid={r['question_id']}  juge={a} contains={b}  gold={r['gold']!r}")
        print(f"   reponse : {(r['predicted'] or '')[:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
