python3 -m venv aqenv
source ./aqenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python3 -m nltk.downloader punkt
python3 ./scripts/download_models.py


# mkdir ../runs
# mkdir ../runs/slurmlogs