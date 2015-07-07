cd $(dirname $0)
wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
tar --extract --file=simple-examples.tgz ./simple-examples/data/ptb.train.txt
tar --extract --file=simple-examples.tgz ./simple-examples/data/ptb.test.txt
tar --extract --file=simple-examples.tgz ./simple-examples/data/ptb.valid.txt

mkdir -p ptb

mv simple-examples/data/ptb.train.txt ./ptb/.
mv simple-examples/data/ptb.test.txt ./ptb/.
mv simple-examples/data/ptb.valid.txt ./ptb/.

rm -r simple-examples*
