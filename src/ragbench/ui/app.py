"""Tableau de bord du banc d'evaluation.

L'interface repond a quatre questions, dans cet ordre :

  1. Qu'est-ce qui a tourne ?          -> onglet Runs
  2. B est-il meilleur que A ?         -> onglet Comparaison
  3. Pourquoi cette question echoue ?  -> onglet Analyse des echecs
  4. Le juge dit-il vrai ?             -> onglet Annotation

L'onglet 3 est le plus important et le plus souvent absent des outils du
marche : une moyenne ne se corrige pas, un echec precis si. L'onglet 4 est
le seul qui demande un humain, et c'est celui qui donne le droit de croire
aux chiffres des autres.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import stats
from . import data

st.set_page_config(page_title="Banc d'evaluation RAG", page_icon="📐", layout="wide")


def _pourcent(x: float) -> str:
    return "—" if pd.isna(x) else f"{x:.3f}"


# =====================================================================
st.title("📐 Banc d'evaluation RAG")

try:
    runs = data.list_runs()
except Exception as exc:  # noqa: BLE001
    st.error(f"Base inaccessible : {exc}")
    st.caption("Verifier que le service `db` tourne, puis lancer `ragbench db init`.")
    st.stop()

if runs.empty:
    st.info(
        "Aucun run enregistre.\n\n"
        "```bash\n"
        "ragbench dataset load multihop-rag --sample 200\n"
        "ragbench run configs/baseline.yml --dataset multihop-rag\n"
        "ragbench eval run 1\n"
        "```"
    )
    st.stop()

tab_runs, tab_compare, tab_failures, tab_annotate, tab_data = st.tabs(
    ["Runs", "Comparaison", "Analyse des echecs", "Annotation", "Corpus"]
)


# =====================================================================
with tab_runs:
    st.subheader("Campagnes")

    display = runs[
        ["id", "label", "config", "dataset", "status", "n_questions", "n_failed", "started_at"]
    ]
    st.dataframe(display, use_container_width=True, hide_index=True)

    run_id = st.selectbox(
        "Detail du run", runs["id"], format_func=lambda i: f"{i} — {runs.set_index('id').loc[i, 'label']}"
    )

    row = runs.set_index("id").loc[run_id]
    for warning in row["warnings"] or []:
        st.warning(warning, icon="⚠️")

    left, right = st.columns([3, 2])

    with left:
        metrics = data.run_metrics(int(run_id))
        if metrics.empty:
            st.info("Aucun score. Lancer `ragbench eval run %d`." % run_id)
        else:
            st.markdown("**Metriques par question** (intervalle de confiance a 95 %, bootstrap)")
            st.dataframe(
                metrics.style.format(
                    {"moyenne": "{:.4f}", "ic_bas": "{:.4f}", "ic_haut": "{:.4f}",
                     "demi_largeur": "{:.4f}"}
                ),
                use_container_width=True, hide_index=True,
            )
            # La demi-largeur est la vraie information de precision : elle dit
            # quel ecart ce run est capable de detecter. Un ecart plus petit
            # qu'elle n'est pas mesurable avec ce nombre de questions.
            worst = metrics.loc[metrics["demi_largeur"].idxmax()]
            st.caption(
                f"Metrique la moins precise : **{worst['metrique']}**, "
                f"± {worst['demi_largeur']:.3f} sur {int(worst['n'])} questions. "
                f"Un ecart inferieur a cette valeur n'est pas detectable ici."
            )

        aggregates = data.run_aggregates(int(run_id))
        if not aggregates.empty:
            st.markdown("**Agregats de run**")
            st.dataframe(
                aggregates.style.format({"valeur": "{:.4f}"}),
                use_container_width=True, hide_index=True,
            )

    with right:
        st.markdown("**Configuration**")
        st.json(data.run_config(int(run_id)), expanded=False)
        st.caption(f"hash `{row['config_hash']}` · code `{row['git_sha'] or 'inconnu'}`")


# =====================================================================
with tab_compare:
    st.subheader("Comparaison appariee de deux runs")
    st.caption(
        "Les deux runs sont compares question par question. C'est ce qui permet "
        "de detecter un ecart de quelques points : la difficulte propre a chaque "
        "question, qui domine la variance, s'annule dans l'ecart."
    )

    col_a, col_b = st.columns(2)
    labels = runs.set_index("id")["label"].to_dict()
    with col_a:
        run_a = st.selectbox("Run A (reference)", runs["id"], index=min(1, len(runs) - 1),
                             format_func=lambda i: f"{i} — {labels[i]}", key="cmp_a")
    with col_b:
        run_b = st.selectbox("Run B", runs["id"], index=0,
                             format_func=lambda i: f"{i} — {labels[i]}", key="cmp_b")

    if run_a == run_b:
        st.info("Choisir deux runs differents.")
    else:
        table = data.comparison_table(int(run_a), int(run_b))
        if table.empty:
            st.warning("Aucune metrique commune aux deux runs.")
        else:
            st.dataframe(
                table.style.format(
                    {"A": "{:.4f}", "B": "{:.4f}", "ecart": "{:+.4f}",
                     "ic_bas": "{:+.4f}", "ic_haut": "{:+.4f}", "p": "{:.3f}"}
                ).map(
                    lambda v: "color: #2e7d32" if v is True else ("color: #999" if v is False else ""),
                    subset=["concluant"],
                ),
                use_container_width=True, hide_index=True,
            )

            n_sig = int(table["concluant"].sum())
            st.caption(
                f"{n_sig} ecart(s) sur {len(table)} dont l'intervalle de confiance exclut zero. "
                f"Attention : avec {len(table)} comparaisons a 5 %, environ "
                f"{len(table) * 0.05:.1f} faux positif(s) sont attendus par hasard seul. "
                "Un ecart isole parmi beaucoup de metriques merite d'etre reteste."
            )

            concluants = table[table["concluant"]]
            if not concluants.empty:
                st.markdown("**Ecarts concluants**")
                for _, r in concluants.iterrows():
                    sens = "meilleur" if r["ecart"] > 0 else "moins bon"
                    st.markdown(
                        f"- `{r['metrique']}` : B est **{sens}** de {abs(r['ecart']):.4f} "
                        f"(IC95 [{r['ic_bas']:+.4f}, {r['ic_haut']:+.4f}], p={r['p']:.3f}, "
                        f"{int(r['B_gagne'])} questions gagnees / {int(r['A_gagne'])} perdues)"
                    )


# =====================================================================
with tab_failures:
    st.subheader("Analyse des echecs")
    st.caption(
        "Une moyenne ne se corrige pas, un echec precis si. Cet onglet sert a "
        "attribuer la faute : retrieval qui n'a pas remonte le bon document, ou "
        "generateur qui n'a pas su l'exploiter."
    )

    run_f = st.selectbox("Run", runs["id"], format_func=lambda i: f"{i} — {labels[i]}", key="fail_run")
    frame = data.predictions(int(run_f))

    if frame.empty:
        st.info("Aucune prediction.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Questions", len(frame))
        col2.metric("Abstentions", int(frame["abstenu"].sum()))
        col3.metric("Erreurs", int(frame["erreur"].notna().sum()))
        col4.metric("Sans reponse (gold)", int(frame["sans_reponse"].sum()))

        score_columns = [c for c in frame.columns if "/" in c]

        # Le tri par defaut isole les cas les plus instructifs : le retrieval a
        # trouve les bons documents mais la reponse est fausse. C'est la que le
        # generateur est en cause, et nulle part ailleurs.
        recall_col = next((c for c in score_columns if c.endswith("recall@5")), None)
        correct_col = next((c for c in score_columns if c.endswith("contains")), None)

        mode = st.radio(
            "Filtre",
            [
                "Tout",
                "Retrieval OK mais reponse fausse (le generateur est en cause)",
                "Retrieval en echec (le retrieval est en cause)",
                "Abstentions injustifiees",
                "Erreurs techniques",
            ],
            horizontal=False,
        )

        view = frame
        if mode.startswith("Retrieval OK") and recall_col and correct_col:
            view = frame[(frame[recall_col] >= 0.99) & (frame[correct_col] < 0.5)]
        elif mode.startswith("Retrieval en echec") and recall_col:
            view = frame[frame[recall_col] < 0.5]
        elif mode.startswith("Abstentions"):
            view = frame[frame["abstenu"] & ~frame["sans_reponse"]]
        elif mode.startswith("Erreurs"):
            view = frame[frame["erreur"].notna()]

        st.caption(f"{len(view)} question(s) sur {len(frame)}")

        columns = ["question_id", "type", "gold", "abstenu"] + score_columns
        st.dataframe(view[columns], use_container_width=True, hide_index=True, height=260)

        if not view.empty:
            qid = st.selectbox("Inspecter la question", view["question_id"])
            record = frame[frame["question_id"] == qid].iloc[0]

            st.markdown(f"**Question** ({record['type']})")
            st.write(record["question"])

            left, right = st.columns(2)
            left.markdown("**Reponse attendue**")
            left.code(record["gold"] or "(aucune)", language=None)
            right.markdown("**Reponse produite**")
            right.code(record["reponse"] or "(vide)", language=None)
            if record["erreur"]:
                st.error(record["erreur"])

            debug = record["_debug"] or {}
            if debug:
                found, missed = len(debug.get("hit", [])), len(debug.get("missed", []))
                st.markdown(f"**Retrieval** — {found} document(s) gold trouve(s), {missed} manque(s)")
                if debug.get("missed"):
                    st.error("Documents gold manques :\n" + "\n".join(f"- {m}" for m in debug["missed"]))

            st.markdown("**Passages remontes**")
            gold_docs = {e["document_external_id"] for e in (record["_gold_evidence"] or [])}
            for ctx in record["_contexts"] or []:
                is_gold = ctx["document_external_id"] in gold_docs
                marker = "✅ gold" if is_gold else "⬜"
                with st.expander(
                    f"{marker} rang {ctx['rank']} · score {ctx['score']:.3f} · "
                    f"{ctx['document_external_id'][:80]}"
                ):
                    st.text(ctx["text"][:2000])


# =====================================================================
with tab_annotate:
    st.subheader("Annotation humaine et validation des juges")
    st.info(
        "**C'est l'ecran qui donne le droit de croire aux autres chiffres.** "
        "Un juge LLM non confronte a un humain ne produit pas une metrique, il "
        "produit une opinion. On annote quelques dizaines de reponses a la main, "
        "puis on calcule le kappa de Cohen entre le juge et l'humain.\n\n"
        "Reperes : < 0,4 faible · 0,4–0,6 moyen · 0,6–0,8 substantiel · > 0,8 excellent.",
        icon="🎯",
    )

    run_h = st.selectbox("Run a annoter", runs["id"],
                         format_func=lambda i: f"{i} — {labels[i]}", key="annot_run")
    frame = data.predictions(int(run_h))

    if frame.empty:
        st.info("Aucune prediction.")
    else:
        from .. import db as _db

        with _db.connect() as conn:
            existing = conn.execute(
                "SELECT question_id, metric, value FROM annotations WHERE run_id = %s",
                (int(run_h),),
            ).fetchall()
        annotated = {(a["question_id"], a["metric"]): a["value"] for a in existing}

        st.caption(f"{len({q for q, _ in annotated})} question(s) deja annotee(s)")

        pending = [
            int(q) for q in frame["question_id"] if (q, "correct") not in annotated
        ]
        target = st.number_input("Objectif d'annotations", 10, 200, 30, step=10)

        if pending:
            qid = pending[0]
            record = frame[frame["question_id"] == qid].iloc[0]
            st.markdown(f"**Question {qid}** ({record['type']})")
            st.write(record["question"])
            col_g, col_p = st.columns(2)
            col_g.markdown("**Attendu**")
            col_g.code(record["gold"] or "(aucune)", language=None)
            col_p.markdown("**Produit**")
            col_p.code(record["reponse"] or "(vide)", language=None)

            st.markdown("**La reponse produite est-elle correcte ?**")
            c1, c2, c3 = st.columns(3)

            def _save(value: float) -> None:
                with _db.connect() as conn:
                    conn.execute(
                        """
                        INSERT INTO annotations (run_id, question_id, metric, value, annotator)
                        VALUES (%s, %s, 'correct', %s, 'human')
                        ON CONFLICT (run_id, question_id, metric, annotator)
                        DO UPDATE SET value = EXCLUDED.value
                        """,
                        (int(run_h), int(qid), value),
                    )
                    conn.commit()
                data.predictions.clear()
                st.rerun()

            if c1.button("✅ Correcte", use_container_width=True):
                _save(1.0)
            if c2.button("❌ Incorrecte", use_container_width=True):
                _save(0.0)
            if c3.button("⏭️ Passer", use_container_width=True):
                _save(-1.0)
        else:
            st.success("Toutes les questions de ce run sont annotees.")

        # --- accord juge / humain -------------------------------------
        if annotated:
            st.divider()
            st.markdown("### Accord entre les juges automatiques et l'annotation humaine")

            human = {q: v for (q, m), v in annotated.items() if m == "correct" and v >= 0}
            score_columns = [c for c in frame.columns if "/" in c]
            rows = []
            for column in score_columns:
                indexed = frame.set_index("question_id")[column].dropna()
                common = sorted(set(human) & set(indexed.index))
                if len(common) < 5:
                    continue
                kappa = stats.cohen_kappa(
                    [human[q] for q in common], [float(indexed.loc[q]) for q in common]
                )
                accord = sum(
                    1 for q in common if round(human[q]) == round(float(indexed.loc[q]))
                ) / len(common)
                rows.append(
                    {"metrique": column, "kappa": kappa, "accord_brut": accord, "n": len(common)}
                )

            if rows:
                table = pd.DataFrame(rows).sort_values("kappa", ascending=False)
                st.dataframe(
                    table.style.format({"kappa": "{:.3f}", "accord_brut": "{:.3f}"}),
                    use_container_width=True, hide_index=True,
                )
                st.caption(
                    "L'accord brut seul est trompeur : sur un jeu ou 80 % des reponses "
                    "sont fausses, un juge qui repond toujours « fausse » atteint 80 % "
                    "d'accord et un kappa de 0. C'est le kappa qui compte."
                )
                if len(human) < target:
                    st.warning(
                        f"{len(human)} annotation(s) sur {target} : le kappa reste instable "
                        "en dessous d'une trentaine d'exemples.",
                        icon="⚠️",
                    )
            else:
                st.info("Pas encore assez d'annotations recoupees avec des scores.")


# =====================================================================
with tab_data:
    st.subheader("Corpus et jeux de questions")
    overview = data.datasets_overview()
    if overview.empty:
        st.info("Aucun dataset. Lancer `ragbench dataset load multihop-rag --sample 200`.")
    else:
        st.dataframe(overview, use_container_width=True, hide_index=True)
        st.caption(
            "`sans_reponse` compte les questions volontairement sans reponse dans le "
            "corpus. Elles sont exclues des metriques de retrieval (leur recall est "
            "indefini, pas nul) et servent a mesurer le negative rejection."
        )
