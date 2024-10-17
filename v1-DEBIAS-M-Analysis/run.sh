pip install DEBIAS-M/.
pip install SBCMP/.

mkdir ../results/HIVRC
mkdir ../results/CRC
mkdir ../results/CRC_with_labels
mkdir ../results/Metabolites
mkdir ../results/Cervix-CIN
mkdir ../results/Cervix-carcinoma
mkdir ../results/Cervix-carcinoma/adaptation
mkdir ../results/Segata
mkdir ../results/Simulations

python main.py
python 01-cervix-roc-plot.py
python 02-Metabolite-plots.py
python 03-adaptation.py
python 04-bcf-analyses.py
python 05-Cervix-RF.py
python 06-make-simulation-plots.py
python 07-Run-immunotherapy.py
