# TF-TabularCondensation

Official implementation of **“C²TC: A Training-Free Framework for Efficient Tabular Data Condensation”**.

---

## 📦 Installation

Follow the steps below to set up TF-TabularCondensation for local development.

### 1) Clone the repo

```bash
git clone https://github.com/Sssara-5/TF-TabularCondensation.git
cd TF-TabularCondensation
```

### 2) Create a Python environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS or Linux:
source .venv/bin/activate
```

### 3) Install dependencies

**TF-TabularCondensation** needs the following requirements to be satisfied beforehand:

- python=3.11  
- numpy=1.26.4  
- pandas=2.2.3  
- scikit-learn=1.7.0  
- pytorch=2.5.1  
- torchvision=0.20.1  
- torchaudio=2.5.1  
- pytorch-cuda=12.4 
- faiss-gpu=1.9.0  

---

## 📂 Datasets

Datasets should be placed in the `dataset/` folder. Each dataset lives in its own subfolder.

Example layout:

```
TF-TabularCondensation
 ├── Appendix
 │    └── Appendix.pdf
 ├── dataset
 │   ├── Adult
 │   │   ├── Adult_preprocessed_info.json
 │   │   ├── Adult_train.csv
 │   │   ├── Adult_val.csv
 │   │   └── Adult_test.csv
 │   └── electricity
 │       ├── electricity_preprocessed_info.json
 │       ├── electricity_train.csv
 │       ├── electricity_val.csv
 │       └── electricity_test.csv
 ├── src
 │   ├── CCTC.py
 │   ├── config.py
 │   ├── eval_syn_ours.py
 │   ├── loader_ours.py
 │   └── loader_syn_ours.py
 └── README.md
```

You can update default paths and options in `src/config.py` if needed.

---

## ✅ Quick Start

Below are example commands to run condensation and evaluation. Replace dataset names and arguments as you like.

### 1) Condense a dataset

**Adult (reduction ratio 0.1 percent)**

```bash
python src/CCTC.py --dataset Adult --reduction 0.001 --out_dir outputs/adult_0_1
```

**electricity (reduction ratio 0.01 percent)**

```bash
python src/CCTC.py --dataset electricity --reduction 0.0001 --out_dir outputs/electricity_1_0
```

Key flags:
- `--dataset` selects the dataset subfolder under `dataset/`
- `--reduction` sets the percentage of the original size kept in the condensed set (for example 0.1 means 0.1 percent)
- `--seed` controls randomness for reproducibility
- `--out_dir` is where condensed data and logs are saved

### 2) Evaluate a condensed set

```bash
python src/eval_syn_ours.py --dataset Adult --syn_dir outputs/adult_0_1
```

This trains standard tabular models on the condensed data and reports metrics on the original test split.

---

## 📑 Appendix

The **Online Appendix** with additional proofs is included in the `Appendix/` folder.


