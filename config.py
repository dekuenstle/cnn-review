from os.path import join

random_seed = 1337
slim_word2vec = True
valid_ratio = 0.1
test_ratio = 0.2
word2vec_dim= 300
max_word_num = 32 # optimal: 19
minibatch_size = 64

base_path = './data'
if slim_word2vec:
    word2vec_file = join(base_path, 'GoogleNews-vectors-negative300-SLIM.bin')
    word2vec_url = "https://github.com/eyaler/word2vec-slim/raw/master/GoogleNews-vectors-negative300-SLIM.bin.gz"
else:
    word2vec_file = join(base_path, 'GoogleNews-vectors-negative300.bin')

polarity_dataset_dir = join(base_path, 'review_polarity', 'txt_sentoken')
data_train_file = join(base_path, 'data_train.npy')
label_train_file = join(base_path, 'label_train.npy')
data_valid_file = join(base_path, 'data_valid.npy')
label_valid_file = join(base_path, 'label_valid.npy')
data_test_file = join(base_path, 'data_test.npy')
label_test_file = join(base_path, 'label_test.npy')
model_file = join(base_path, 'model.h5')

port = 80
