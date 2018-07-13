TRAIN_DATA_DIR = /project/piqasso/Collection/DMD/2018/DMD-2018-sharedtask
TRAIN_TASK_1 = $(TRAIN_DATA_DIR)/Task-1/training/train.csv
TRAIN_TASK_2 = $(TRAIN_DATA_DIR)/Task-2/training/train.csv

GPU = 1

EMBEDDING_DIM=50
BATCH_SIZE=256
EPOCHS=1
LSTM_LAYER_SIZE=128
DROPOUT=0.2
BIDIRECTIONAL=-bi
NODENSE=-nodense

ed$(EMBEDDING_DIM)b$(BATCH_SIZE)lstm$(LSTM_LAYER_SIZE)dout$(DROPOUT)ep$(EPOCHS)$(BIDIRECTIONAL)$(NODENSE).model: $(TRAIN_TASK_1)
	export CUDA_VISIBLE_DEVICES="$(GPU)"; python classifier.py train -ed $(EMBEDDING_DIM) -b $(BATCH_SIZE) -e $(EPOCHS) -lstm $(LSTM_LAYER_SIZE) -d $(DROPOUT) $(BIDIRECTIONAL) $(NODENSE) -f $@ < $<


botnet_ed$(EMBEDDING_DIM)b$(BATCH_SIZE)lstm$(LSTM_LAYER_SIZE)dout$(DROPOUT)ep$(EPOCHS)$(BIDIRECTIONAL)$(NODENSE).model: $(TRAIN_TASK_2)
	export CUDA_VISIBLE_DEVICES="$(GPU)"; head -300 $< | python classifier.py train -ed $(EMBEDDING_DIM) -b $(BATCH_SIZE) -e $(EPOCHS) -lstm $(LSTM_LAYER_SIZE) -d $(DROPOUT) $(BIDIRECTIONAL) $(NODENSE) -nb 21 -f $@


predict:
	export CUDA_VISIBLE_DEVICES="$(GPU)"; shuf -n 100 $(TRAIN_TASK_1) | python classifier.py predict --file-model 1531224509.model
