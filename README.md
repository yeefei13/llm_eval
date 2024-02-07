# RAG Framework

## Setup

Create a new conda environment and install the required packages:

```bash
conda create --name rag python=3.10
conda activate rag
pip install -r requirements.txt
```

Download the data using the download script provided:

```bash
cd data/datasets/natural-questions
sh download.sh
```

Run the RAG app:
```bash
python apps/rag.py
```

You can build your own apps by following the example in `apps/rag.py`.