# Generator Data lab

## Basic structure
- src/
  - download
  - transform (combination of cleaning and reformatting)
  - explore
  - analysis
  - util (AWS packages, navigators, etc.)
- data/
  - raw
  - tmp (hold pickled files)
  - clean
  - final
- figures/ (follows data)
  - raw
  - clean
  - final

canonical structure will remain
- src/
- data/
- figures/
- Make
- .gitignore
- README.md

## Naming conventions
- use all lower case
- date follows YYYYMMDD
- use underscores to separate words

## Prompting
- [ ] Level of intensity
  - [ ] basic exploration
    - simple scripting
    - single file in src folder
    - empty (with placeholder) data folder
    - empty (with placeholder) figures folder
  - [ ] exploratory analysis, self-catered data
    - a few sample datasets
    - src/transform, explore, analyze
    - data/raw, tmp, clean, final
    - figures
  - [ ] large, cloud, downloaded data
    - [ ] Ask for AWS bucket-name, or other relevant information
    - src/download, transform, explore, analyze
    - data/raw, tmp, clean, final
    - figures

- [ ] Util will always be provided
  - [ ] custom path-finder
  - [ ] AWS downloading scripts (also used as examples)
  - [ ] Lambda scrips, invoke

- [ ] Packages in all files
  - [ ] numpy
  - [ ] pandas
  - [ ] matplotlib
  - [ ] scipy (in explore and analysis files)
  - [ ] scikit-learn (only in explore and analysis files)

## Coding paradigms
- Context manager for path finding
