MODEL='FarkasNet101' #FarkasNet model goes here; see models/farkas.py for available options

# Setup
TIMESTAMP=`date +%y-%m-%dT%H%M%S`  # Use this in LOGDIR
DATASET='cifar10'   # Use the dataset name in LOGDIR
DATADIR='/PATH/TO/data/'  # Shared data file store

BASELOG='./logs/'$DATASET/$MODEL
LOGDIR=$BASELOG/'farkaslayers-'$TIMESTAMP
SCRATCH='/mnt/data/scratch/'$USER-'runs/'$TIMESTAMP  # During training write to a local drive, not a network drive

mkdir -p $DATADIR
mkdir -p $SCRATCH
chmod g+rwx $SCRATCH # so that others can delete this folder if we kill the experiment and forget to
mkdir -p $BASELOG

ln -s $SCRATCH $LOGDIR

#Enable cuda
CUDA_VISIBLE_DEVICES=1 \
python -u ./train.py \
    --no-bn \ 
    --bias \
    --weight-init standard \
    --zero-last \
    --lr 1e-1 \
    --cutout 16 \
    --model $MODEL \
    --dataset $DATASET \
    --datadir $DATADIR \
    --logdir $LOGDIR \
    | tee $LOGDIR/log.out 2>&1 # Write stdout directly to log.out.
                               # If you don't want to see the output, replace
                               # '| tee' with '>'
                               # (with '>', if you want to see results in real time,
                               # use tail -f on the logfile)

rm $LOGDIR
mv $SCRATCH $LOGDIR
