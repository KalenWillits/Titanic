
# Kaggle Competition
## Titanic: Machine Learning from Disaster

In this project we attempt to predict who survives the disaster of the Titanic.
The data and rules for the competition are found at https://www.kaggle.com/c/titanic.




```
sqlite3 < build_db.sh
```

.
├── build_db.py
├── build_db.sh
├── data
│   ├── gender_submission.csv
│   ├── test.csv
│   ├── titanic-backup.zip
│   ├── titanic-prediction.csv
│   ├── titanic.sqlite
│   └── train.csv
├── docs
│   ├── blog.md
│   ├── EDA.md
│   └── model_metrics.md
├── figures
│   ├── distrobution-of-fare-prices.png
│   └── heatmap.png
├── library.py
├── main.py
├── models
│   └── svc.pkl
├── notebook.ipynb
├── __pycache__
│   ├── library.cpython-37.pyc
│   └── library.cpython-38.pyc
├── readme.md
├── sandbox
│   └── sandbox.py
├── sources
│   └── links.md
├── titanic.sqlite
└── tree.txt

7 directories, 24 files

Package                Version
---------------------- -------------------
astroid                2.4.2
autopep8               1.5.4
backcall               0.2.0
certifi                2020.11.8
chardet                3.0.4
cycler                 0.10.0
decorator              4.4.2
flake8                 3.8.4
idna                   2.10
ipykernel              5.3.4
ipython                7.19.0
ipython-genutils       0.2.0
isort                  5.6.4
jedi                   0.17.2
joblib                 0.17.0
jupyter-client         6.1.7
jupyter-core           4.6.3
kaggle                 1.5.9
kiwisolver             1.3.1
lazy-object-proxy      1.4.3
matplotlib             3.3.3
mccabe                 0.6.1
numpy                  1.19.4
pandas                 1.1.4
parso                  0.7.1
pexpect                4.8.0
pickleshare            0.7.5
Pillow                 8.0.1
pip                    20.2.4
pluggy                 0.13.1
prompt-toolkit         3.0.8
ptyprocess             0.6.0
pycodestyle            2.6.0
pydocstyle             5.1.1
pyflakes               2.2.0
Pygments               2.7.2
pylint                 2.6.0
pyparsing              2.4.7
python-dateutil        2.8.1
python-jsonrpc-server  0.4.0
python-language-server 0.36.1
python-slugify         4.0.1
pytz                   2020.4
pyzmq                  19.0.2
requests               2.24.0
rope                   0.18.0
scikit-learn           0.23.2
scipy                  1.5.4
seaborn                0.11.0
setuptools             50.3.1.post20201107
six                    1.15.0
slugify                0.0.1
snowballstemmer        2.0.0
SQLAlchemy             1.3.20
tabulate               0.8.7
text-unidecode         1.3
threadpoolctl          2.1.0
toml                   0.10.2
tornado                6.1
tqdm                   4.51.0
traitlets              5.0.5
ujson                  4.0.1
urllib3                1.25.11
wcwidth                0.2.5
wheel                  0.35.1
wrapt                  1.12.1
yapf                   0.30.0
