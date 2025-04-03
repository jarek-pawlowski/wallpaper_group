# wallpaper_group
### generator of 2D lattices falling into different wallpaper groups

To generate lattices (and their FFTs) just adjust *no_samples_to_be_generated* in `lattice.py` and run:
```python
python lattice.py
```
- labels (= group names) will be stored in `./generated/labels.json`

Quasicrystals.py + Qresults -> 1d incommensurate chain in Fourier space 

1Dmodes.py + 1Dmodes_results ->  1d (in)commensurate chain in Fourier space
1Dmodes_.py -> (in)commensurate chain in Fourier space, phason branches

2Dmodes.py -> 12-fold quasicrystal

Prepresentation.py -> Phase space representation for 2D wallpapers, then symmetry detector
Quasicrystals.py -> Experiments with phase-function for 1D quasicrystals