mkdir "data"
mkdir "confusion_matrices"
mkdir "images"
mkdir "bert_embeddings"

cd "data"

# download data
wget 'http://archive.aueb.gr:8085/files/en_partut-ud-train.conllu' 'en_partut-ud-train.conllu.txt'
wget 'http://archive.aueb.gr:8085/files/en_partut-ud-test.conllu' 'en_partut-ud-test.conllu.txt'
wget 'http://archive.aueb.gr:8085/files/en_partut-ud-dev.conllu' 'en_partut-ud-dev.conllu.txt'

sudo apt-get install --reinstall libxcb-xinerama0
cd ..
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt