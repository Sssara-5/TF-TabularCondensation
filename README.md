# **C²TC**-TabularCondensation

Official implementation of **“C²TC: A Training-Free Framework for Efficient Tabular Data Condensation”**.

---

## 📦 Installation

Follow the steps below to set up TF-TabularCondensation for local development.

### 1) Clone the repo

```bash
git clone https://github.com/Sssara-5/TF-TabularCondensation.git
cd TF-TabularCondensation
```

### 2) Environment (Conda recommended)

```bash
conda env create -f environment.yml
conda activate cctc
```

---

## 🚀 Quick start

```bash
conda activate cctc
bash run_pipeline.sh
```

Or without activating the shell:

```bash
conda run -n cctc --no-capture-output ./run_pipeline.sh
```

Run the above from the directory that contains `environment.yml` and `run_pipeline.sh`. After activating `cctc`, you can sanity-check dependencies and GPU visibility:

```bash
python3 verify_env.py
```

---

## 📖 Citation

Paper: [arXiv:2602.21717](https://arxiv.org/abs/2602.21717)

```bibtex
@article{xu2026c,
  title   = {{C$^2$TC}: A Training-Free Framework for Efficient Tabular Data Condensation},
  author  = {Xu, Sijia and Li, Fan and Wang, Xiaoyang and Yang, Zhengyi and Lin, Xuemin},
  journal = {arXiv preprint arXiv:2602.21717},
  year    = {2026}
}
```

