DATAROOT=../instant-ngp-scaffold/results/grayscale
convert() {
    python scripts/nerfout2dual.py \
        --dataroot=data/$1 \
        --hrsrcroot=$2 \
        --grsrcroot=$3 \
        --name=nerfout
}
# convert coffee_martini-kmeans-6 $DATAROOT/coffee_martini-color-frame1-base $DATAROOT/coffee_martini-gray-frame1-base/kmeans-6 # debug
traindual() {
    python code/train.py \
        --dataroot=data/$1 \
        --scale=$2 \
        --arch=nerfrt4ksr_rep \
        --benchmark=nerfoutdual_train \
        --checkpoint-id=$1/nerfrt4ksr_x"$2"
}
# traindual coffee_martini-kmeans-6 2 # debug
testdual() {
    python code/test.py \
        --dataroot=data/$1 \
        --scale=$2 \
        --arch=nerfrt4ksr_rep \
        --benchmark=nerfoutdual \
        --checkpoint-id=$1/nerfrt4ksr_x"$2"_rep_model \
        --save-results=results/$1-nerfoutdual-x"$2".json
}
# testdual coffee_martini-kmeans-6 2 # debug
trainsingle() {
    python code/train.py \
        --dataroot=data/$1 \
        --scale=$2 \
        --arch=rt4ksr_rep \
        --benchmark=nerfout_train \
        --checkpoint-id=$1/rt4ksr_x"$2"
}
# trainsingle coffee_martini-kmeans-6 2 # debug
testsingle() {
    python code/test.py \
        --dataroot=data/$1 \
        --scale=$2 \
        --arch=rt4ksr_rep \
        --benchmark=nerfout \
        --checkpoint-id=$1/rt4ksr_x"$2"_rep_model \
        --save-results=results/$1-nerfout-x"$2".json
}
# testsingle coffee_martini-kmeans-6 2 # debug
doboth() {
    traindual $1 $2
    testdual $1 $2
    trainsingle $1 $2
    testsingle $1 $2
}
doall() {
    convert $1-kmeans-$2 $DATAROOT/$1-color-frame1-base $DATAROOT/$1-gray-frame1-base/kmeans-$2
    doboth $1-kmeans-$2 2
    doboth $1-kmeans-$2 3
    doboth $1-kmeans-$2 4
}
command() {
    doall $1 6
    doall $1 7
    doall $1 8
    doall $1 9
    doall $1 10
}
command coffee_martini
