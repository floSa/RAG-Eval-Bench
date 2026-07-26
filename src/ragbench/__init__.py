"""ragbench — banc d'evaluation RAG on-premise.

Couches, du bas vers le haut :

  settings / config   parametres d'infra vs parametres d'experience
  llm / db            acces au serveur d'inference et a Postgres/pgvector
  rag/                le pipeline evalue (chunking, retrieval, generation)
  datasets/           chargement des corpus et jeux de questions
  evaluators/         les plugins de mesure, un par framework
  runner/             orchestration des campagnes
  ui/                 tableau de bord de comparaison

Regle de dependance : une couche ne connait que celles du dessous.
"""

__version__ = "0.1.0"
