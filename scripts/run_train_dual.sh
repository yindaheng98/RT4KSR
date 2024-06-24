traindual() {
    if [ -e "code/checkpoints/$1/nerfrt4ksr_x$2.pth" ]; then
        echo "skip $1/nerfrt4ksr_x$2"
        return 0
    fi
    python code/train.py \
        --dataroot=data/$1 \
        --scale=$2 \
        --arch=nerfrt4ksr_rep \
        --benchmark=nerfoutdual_train \
        --checkpoint-id=$1/nerfrt4ksr_x"$2" \
        --epoch $3 \
        --batch-size 16 \
        --crop-size 298 \
        --num-workers 16
}
# traindual coffee_martini-kmeans-16-scale-12-rot-10-f_dc-6-f_rest-6-opacity-6 2 # debug
testdual() {
    if [ -e "srresults/$1-nerfoutdual-x$2.json" ]; then
        echo "skip srresults/$1-nerfoutdual-x$2.json"
        return 0
    fi
    python code/test.py \
        --dataroot=data/$1 \
        --scale=$2 \
        --arch=nerfrt4ksr_rep \
        --benchmark=nerfoutdual \
        --checkpoint-id=$1/nerfrt4ksr_x"$2"_rep_model \
        --save-results=srresults/$1-nerfoutdual-x"$2".json
}
trainsingle() {
    if [ -e "code/checkpoints/$1/rt4ksr_x$2.pth" ]; then
        echo "skip $1/rt4ksr_x$2"
        return 0
    fi
    python code/train.py \
        --dataroot=data/$1 \
        --scale=$2 \
        --arch=rt4ksr_rep \
        --benchmark=nerfout_train \
        --checkpoint-id=$1/rt4ksr_x"$2" \
        --epoch $3 \
        --batch-size 32 \
        --crop-size 298 \
        --num-workers 16
}
# trainsingle coffee_martini-kmeans-16-scale-12-rot-10-f_dc-6-f_rest-6-opacity-6 2 # debug
testsingle() {
    if [ -e "srresults/$1-nerfout-x$2.json" ]; then
        echo "skip srresults/$1-nerfout-x$2.json"
        return 0
    fi
    python code/test.py \
        --dataroot=data/$1 \
        --scale=$2 \
        --arch=rt4ksr_rep \
        --benchmark=nerfout \
        --checkpoint-id=$1/rt4ksr_x"$2"_rep_model \
        --save-results=srresults/$1-nerfout-x"$2".json
}
# testsingle coffee_martini-kmeans-16-scale-12-rot-10-f_dc-6-f_rest-6-opacity-6 2 # debug
doboth() {
    traindual $1 $2 64
    testdual $1 $2 64
    trainsingle $1 $2 64
    testsingle $1 $2 64
}
doall() {
    trainsingle $1-kmeans-$2 1 64
    testsingle $1-kmeans-$2 1 64
    doboth $1-kmeans-$2 2
    doboth $1-kmeans-$2 3
    doboth $1-kmeans-$2 4
}
doall_wrap() {
    doall $1 qp-$7-scale-$2-rot-$3-f_dc-$4-f_rest-$5-opacity-$6
}
command() {
    doall_wrap $1  8  4  4  4 4 8
    doall_wrap $1 16 16 16 16 4 8
    doall_wrap $1 14 13 13 13 4 8
    doall_wrap $1 12 10 10 10 4 8
    doall_wrap $1 10  7  7  7 4 8
}
# command stnerf-taekwondo
# command stnerf-walking

command coffee_martini
command cook_spinach
command cut_roasted_beef
command flame_salmon_1
command flame_steak
command sear_steak

command discussion
command stepin
command trimming
command vrheadset
