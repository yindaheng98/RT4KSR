ARCHSR=srresnet
ARCHDUAL=nerfsrresnet
traindual() {
    if [ -e "code/checkpoints/$1/${ARCHDUAL}_x1.pth" ]; then
        echo "skip $1/${ARCHDUAL}_x1"
        return 0
    fi
    # echo \
    python code/train.py \
        --dataroot=data/$1 \
        --scale=1 \
        --arch=$ARCHDUAL \
        --benchmark=nerfoutdual_train \
        --checkpoint-id=$1/${ARCHDUAL}_x1 \
        --epoch $2 \
        --batch-size 8 \
        --crop-size $3 \
        --num-workers 16
}
# traindual coffee_martini-kmeans-qp-none-scale-16-rot-16-f_dc-16-f_rest-16-opacity-4-warped 64 1176 # debug
testdual() {
    if [ -e "srresults/$1-nerfoutdual-x1.json" ]; then
        echo "skip srresults/$1-nerfoutdual-x1.json"
        return 0
    fi
    # echo \
    python code/test.py \
        --dataroot=data/$1 \
        --scale=1 \
        --arch=$ARCHDUAL \
        --benchmark=nerfoutdual \
        --checkpoint-id=$1/${ARCHDUAL}_x1_rep_model \
        --save-results=srresults/$1-nerfoutdual-x1.json
}
# testdual coffee_martini-kmeans-qp-none-scale-16-rot-16-f_dc-16-f_rest-16-opacity-4-warped
doboth() {
    traindual $1 64 $2
    testdual $1
}
# doboth coffee_martini-kmeans-qp-none-scale-16-rot-16-f_dc-16-f_rest-16-opacity-4-warped 1176
doall() {
    doboth $1-kmeans-$2-warped $3
    doboth $1-kmeans-$2-warpednoee $3
}
doall_wrap() {
    doall $1 qp-none-scale-$3-rot-$4-f_dc-$5-f_rest-$6-opacity-$7 $2
}
doall_wrap coffee_martini 1176 16 16 16 16 4
