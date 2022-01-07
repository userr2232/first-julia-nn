# FIRST-JULIA-NN

1. Get the full dataset inside the `data` directory. (Data_Julia subdir and Completed_SAO_V3.hdf5 file)
2. Install requirements
```
pip install -r requirements.txt
```
3. Create partitions
```
python src/main.py root=(pwd)
```
4. Generate folds and run study
```
python src/main.py root=(cwd) action=run_study
```