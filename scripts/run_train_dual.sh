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
traindual coffee_martini-kmeans-qp-none-scale-16-rot-16-f_dc-16-f_rest-16-opacity-4-warped 64 1176 # debug
