# TO DO LIST

## Visualization

- [x] Distribution of particles/jets raw features (see `eda/particle_eda.ipynb` and `eda/jet_eda.ipynb`)
- [ ] Distribution of particles preprocessed features
- [ ] 3D visualization of an event with two jets 
- [ ] Figures representing the preprocessed features (powerpoint, tikz, other ?)
    - delta angles
    - projected momenta
- [ ] Figures representing the architecture of the neural networks
- [ ] Results of the neural networks 
    - ROC curves
    - confusion matrices
    - loss curves
    - accuracy curves
    - other ?

## Preprocessing

- [x] ROOT to CSV conversion and feature extraction (see `preprocessing/make_dataset.py`)
- [ ] Compute particle quantities w.r.t. jet quantities
  - normalized energy $\hat{E}_{p} = E_p / E _j \in[0,\,1]$ then scaled to $E=\hat{E}_{p} - \hat{E}_{\text{average}} \in [-\hat{E}_{\text{average}},\,1-\hat{E}_{\text{average}}]\sim$ centered around 0
  - momentum components w.r.t. jet direction
  - delta angles between particle and jet
  - other ?
- [ ] PCA / dim. reduction on the entire set of raw features to see what comes out

## Machine learning

- [ ] What architectures?
- [ ] Train architectures on raw features
- [ ] Train architectures on preprocessed features
- [ ] How do we evaluate the performance of nns?

## Paper

- [ ] Abstract (the last thing to do)
- [ ] Introduction (we begin with it and then redo it at the end)
- [ ] Related work &rarr; we do an overview of the different methods used in the literature 
- [ ] Dataset &rarr; we describe the dataset 
- [ ] Preprocessing &rarr; we describe the preprocessing steps and why we chose them
- [ ] Model &rarr; we describe the machine learning model(s) in great detail
- [ ] Results 
- [ ] Conclusions

----

polar E = mass
polar PX = pt
polar PY = eta
polar PZ = phi