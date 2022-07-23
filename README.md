# Isla Mcts
Library developed for the [Isla Research Group](https://www.di.univr.it/?ent=grupporic&id=451) at University of Verona
## Implemeted Algorithms:

- Montecarlo Tree Search (UCT) [1]
- Montecarlo Tree Search with State Progressive Widening [2]
- Montecarlo Tree Search with Action Progressive Widening [2]
- Montecarlo Tree Search with Double Progressive Widening [2]
- Montecarlo Tree Search with Voronoi Progressive Widening - WORK IN PROGRESS [3]

## Requirements

1. Install libraries listed in requirements.txt using `pip install -r requirements.txt`
2. Install [graphviz](https://graphviz.org/download/)
3. Install goddard environment
    1. Download and unzip [goddard repository](https://github.com/osannolik/gym-goddard) into main directory
    2. Install library<br/>`cd gym-goddard`<br/>`pip install -e .`

## Reference

- [1] [Kocsis, L., Szepesvári, C. (2006). Bandit Based Monte-Carlo Planning. In: Fürnkranz, J., Scheffer, T., Spiliopoulou, M. (eds) Machine Learning: ECML 2006. ECML 2006. Lecture Notes in Computer Science(), vol 4212. Springer, Berlin, Heidelberg](https://doi.org/10.1007/11871842_29)
- [2] [Adrien Cou toux, Jean-Baptiste Hoock, Nataliya Sokolovska, Olivier Teytaud, Nicolas Bonnard. Continuous Upper Confidence Trees. LION’11: Proceedings of the 5th International Conference on Learning and Intelligent OptimizatioN, Jan 2011, Italy. pp.TBA. hal-00542673v2](https://hal.archives-ouvertes.fr/hal-00542673)
- [3] [Lim, Michael H. Voronoi Progressive Widening: Efficient Online Solvers for Continuous Space MDPs and POMDPs with Provably Optimal Components.](https://doi.org/10.48550/arXiv.2012.10140)