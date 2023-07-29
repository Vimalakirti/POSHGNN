# POSHGNN
The source code for POSHGNN (ICDE 2024).

## Environment
```
conda env create -f environment.yml
conda activate mr_rec
```

## Run
Please run the following script based on the provided toy dataset (`dataset/example.csv`):
```
python train.py
```

## Complexity analysis of each module in POSHGNN (per time step)
Given $N$ users surrounding our target user at time step $t$ and all their occluded relations $\mathcal{E}_t$ (described in `full.pdf`), we provide the corresponding complexity analysis for each module with these notations.
- MIA (Multi-modal information aggregator)'s time complexity is $\mathcal{O}(N+\mathcal{E}_t)$.
- PDR (Partial View De-occlusion Recommender)'s time complexity is $\mathcal{O}(\mathcal{E}_t)$.
- LWP (Learning Which to Preserve)'s time complexity is
$\mathcal{O}(\mathcal{E}_t)$.