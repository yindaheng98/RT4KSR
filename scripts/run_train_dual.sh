ARCHSR=srresnet
ARCHDUAL=nerfsrresnet
traindual() {
    if [ -e "code/checkpoints/$1/${ARCHDUAL}_x$2.pth" ]; then
        echo "skip $1/${ARCHDUAL}_x$2"
        return 0
    fi
    # echo \
    python code/train.py \
        --dataroot=data/$1 \
        --scale=$2 \
        --arch=$ARCHDUAL \
        --benchmark=nerfoutdual_train \
        --checkpoint-id=$1/${ARCHDUAL}_x"$2" \
        --epoch $3 \
        --batch-size 8 \
        --crop-size $4 \
        --num-workers 16
}
# traindual coffee_martini-kmeans-16-scale-12-rot-10-f_dc-6-f_rest-6-opacity-6 2 # debug
testdual() {
    if [ -e "srresults/$1-nerfoutdual-x$2.json" ]; then
        echo "skip srresults/$1-nerfoutdual-x$2.json"
        return 0
    fi
    # echo \
    python code/test.py \
        --dataroot=data/$1 \
        --scale=$2 \
        --arch=$ARCHDUAL \
        --benchmark=nerfoutdual \
        --checkpoint-id=$1/${ARCHDUAL}_x"$2"_rep_model \
        --save-results=srresults/$1-nerfoutdual-x"$2".json
}
trainsingle() {
    if [ -e "code/checkpoints/$1/${ARCHSR}_x$2.pth" ]; then
        echo "skip $1/${ARCHSR}_x$2"
        return 0
    fi
    # echo \
    python code/train.py \
        --dataroot=data/$1 \
        --scale=$2 \
        --arch=$ARCHSR \
        --benchmark=nerfout_train \
        --checkpoint-id=$1/${ARCHSR}_x"$2" \
        --epoch $3 \
        --batch-size $((4*$2)) \
        --crop-size $4 \
        --num-workers 16
}
# trainsingle coffee_martini-kmeans-16-scale-12-rot-10-f_dc-6-f_rest-6-opacity-6 2 # debug
testsingle() {
    if [ -e "srresults/$1-nerfout-x$2.json" ]; then
        echo "skip srresults/$1-nerfout-x$2.json"
        return 0
    fi
    # echo \
    python code/test.py \
        --dataroot=data/$1 \
        --scale=$2 \
        --arch=$ARCHSR \
        --benchmark=nerfout \
        --checkpoint-id=$1/${ARCHSR}_x"$2"_rep_model \
        --save-results=srresults/$1-nerfout-x"$2".json
}
# testsingle coffee_martini-kmeans-16-scale-12-rot-10-f_dc-6-f_rest-6-opacity-6 2 # debug
doboth() {
    traindual $1 $2 64 $3
    testdual $1 $2
    trainsingle $1 $2 64 $3
    testsingle $1 $2
}
doall() {
    trainsingle $1-kmeans-$2 1 64 $3
    testsingle $1-kmeans-$2 1 64 $3
    # doboth $1-kmeans-$2 2 $3
    # doboth $1-kmeans-$2 3 $3
    doboth $1-kmeans-$2 4 $3
}
doall_wrap() {
    doall $1 qp-$3-scale-$4-rot-$5-f_dc-$6-f_rest-$7-opacity-$8 $2
}
command() {
    doall_wrap $1 $2 8  8  4  4  4 4
    doall_wrap $1 $2 8 16 16 16 16 4
    doall_wrap $1 $2 8 14 13 13 13 4
    doall_wrap $1 $2 8 12 10 10 10 4
    doall_wrap $1 $2 8 10  7  7  7 4
}
# command stnerf-taekwondo
# command stnerf-walking

command coffee_martini 1176
command cook_spinach 1176
command cut_roasted_beef 1176
command flame_salmon_1 1176
command flame_steak 1176
command sear_steak 1152

command discussion 672
command stepin 672
command trimming 672
command vrheadset 672
