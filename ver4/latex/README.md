# ver4 paper

Main source: `preference_transition_mamba_rl.tex`

The report contains:

- the hybrid Mamba/LightGCN and multi-agent method;
- latent preference prototype discovery;
- current-to-next preference transition prediction;
- four preference auxiliary losses and the final ranking equation;
- staged supervised/RL optimization and full-catalog evaluation;
- protocol-aware published MovieLens, Yelp, and Amazon result tables;
- the completed ver4 Beauty result available at generation time;
- separation of ver4 measurements, ver3 history, and external results.

Run `bash build_latex.sh` with a full TeX Live installation. TikZ/PGF is
required for the two architecture figures.
