mkdir "data"
#curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3226{/ud-treebanks-v2.6.tgz,/ud-documentation-v2.6.tgz,/ud-tools-v2.6.tgz}
cd "data"
#curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1743{/deltacorpus-1.1.tar}

# download data
wget 'http://archive.aueb.gr:8085/files/en_partut-ud-train.conllu' 'en_partut-ud-train.conllu.txt'
wget 'http://archive.aueb.gr:8085/files/en_partut-ud-test.conllu' 'en_partut-ud-test.conllu.txt'
wget 'http://archive.aueb.gr:8085/files/en_partut-ud-dev.conllu' 'en_partut-ud-dev.conllu.txt'
