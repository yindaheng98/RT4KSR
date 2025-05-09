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
merge() {
    link() {
        mkdir -p $2
        for i in $(ls $1); do
            if [ -e "$2/$i.$3.png" ]; then
                rm $2/$i.$3.png
            fi
            ln -s $PWD/$1/$i $2/$i.$3.png
        done
    }
    linkdataset() {
        link $1/Gray $2/Gray $3
        link $1/HR $2/HR $3
        link $1/LR_bicubic_x1 $2/LR_bicubic_x1 $3
    }
    linkbothset() {
        linkdataset $1/testsets/nerfout/train $2/testsets/nerfout/train $3
        linkdataset $1/testsets/nerfout/val $2/testsets/nerfout/val $3
    }
    linkbothset data/$1-kmeans-$2-warped data/$1-kmeans-merge-warped $2
}
# merge coffee_martini qp-none-scale-16-rot-16-f_dc-16-f_rest-16-opacity-4
doall() {
    doboth $1-kmeans-$2-nowarp $3
    doboth $1-kmeans-$2-warped $3
    doboth $1-kmeans-$2-warpednoee $3
    merge $1 $2
}
doall_wrap() {
    doall $1 qp-none-scale-$3-rot-$4-f_dc-$5-f_rest-$6-opacity-$7 $2
}
# doall_wrap coffee_martini 1176 16 16 16 16 4
command() {
    doall_wrap $1 $2  8  4  4  4 4
    doall_wrap $1 $2 16 16 16 16 4
    doall_wrap $1 $2 14 13 13 13 4
    doall_wrap $1 $2 12 10 10 10 4
    doall_wrap $1 $2 10  7  7  7 4
    doboth $1-kmeans-merge-warped $2
}

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

command taekwondo 904
command walking 846
