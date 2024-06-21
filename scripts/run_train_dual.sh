traindual() {
    python code/train.py \
        --dataroot=data/$1 \
        --scale=$2 \
        --arch=nerfrt4ksr_rep \
        --benchmark=nerfoutdual_train \
        --checkpoint-id=$1/nerfrt4ksr_x"$2"
}
# traindual coffee_martini-kmeans-16-scale-12-rot-10-f_dc-6-f_rest-6-opacity-6 2 # debug
testdual() {
    python code/test.py \
        --dataroot=data/$1 \
        --scale=$2 \
        --arch=nerfrt4ksr_rep \
        --benchmark=nerfoutdual \
        --checkpoint-id=$1/nerfrt4ksr_x"$2"_rep_model \
        --save-results=srresults/$1-nerfoutdual-x"$2".json
}
traindual_colordecay() {
    python code/train.py \
        --dataroot=data/$1 \
        --scale=$2 \
        --arch=nerfrt4ksr_rep \
        --benchmark=nerfoutdual_train_colordecay \
        --checkpoint-id=$1/nerfrt4ksr_x"$2"_colordecay
}
# traindual_colordecay coffee_martini-kmeans-16-scale-12-rot-10-f_dc-6-f_rest-6-opacity-6 2 # debug
testdual_colordecay() {
    python code/test.py \
        --dataroot=data/$1 \
        --scale=$2 \
        --arch=nerfrt4ksr_rep \
        --benchmark=nerfoutdual_colordecay \
        --checkpoint-id=$1/nerfrt4ksr_x"$2"_colordecay_rep_model \
        --save-results=srresults/$1-nerfoutdual_colordecay-x"$2".json
}
# testdual_colordecay coffee_martini-kmeans-16-scale-12-rot-10-f_dc-6-f_rest-6-opacity-6 2 # debug
trainsingle() {
    python code/train.py \
        --dataroot=data/$1 \
        --scale=$2 \
        --arch=rt4ksr_rep \
        --benchmark=nerfout_train \
        --checkpoint-id=$1/rt4ksr_x"$2"
}
# trainsingle coffee_martini-kmeans-16-scale-12-rot-10-f_dc-6-f_rest-6-opacity-6 2 # debug
testsingle() {
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
    traindual $1 $2
    testdual $1 $2
    traindual_colordecay $1 $2
    testdual_colordecay $1 $2
    trainsingle $1 $2
    testsingle $1 $2
}
doall() {
    trainsingle $1-kmeans-$2 1
    testsingle $1-kmeans-$2 1
    doboth $1-kmeans-$2 2
    doboth $1-kmeans-$2 3
    doboth $1-kmeans-$2 4
}
command() {
    doall $1 16-scale-12-rot-10-f_dc-6-f_rest-6
    # TODO: more
}
command stnerf-taekwondo
command stnerf-walking
command coffee_martini
command flame_steak
command sear_steak
command discussion
command stepin
command trimming
command vrheadset
