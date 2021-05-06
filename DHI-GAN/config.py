

'''
all configs for model training

'''

class Config(object):
    env = "comp"
    gpu = '0'
    backbone = "densenet"
    classify = "softmax"
    num_classes = 9490 #只能和类别数相等
    metric = "add_margin"#111
    easy_margin = False
    use_se = False
    loss ="CrossEntropyLoss" #focal_loss

    display = False
    write = False

    train_list = r"../TMI_TI/label/train.txt"

    FilenameXT = r"../TMI_TI/label/testXT.txt"
    FilenameXR = r"../TMI_TI/label/testXR.txt"
    #Far_list = r"E:\Tooth_Data\data\PytorchImg\VerFar.txt"
    
    FirstFile = r"/home/wiseatc/lyi/torch/{}/".format(metric)
    
    checkpoints_path = FirstFile + r"/savefile/checkpoints"
    test_model_list = FirstFile + r"/savefile/checkpoints"

    pretrained = False
    continue_train = False
    load_model_path = FirstFile + r"/savefile/pretrained/resnet18_25.pth"

    save_interval = 1 #每隔10个epoch保存一次
    train_batch_size = 16 #batch size
    test_batch_size = 4

    input_shape = (1,128,128)
    optimizer = "Adam"
    num_workers = 4 # how many workers for loading data
    print_freq = 100 #print info every N batch

    max_epoch = 100
    lr = 1e-5 #initial learning rate
    lr_step = 50
    lr_decay = 0.1 # when val_loss increase , lr = lr * lr_decay
    weight_decay = 1e-4
    momentum = 0.9

    debug_file = "/tmp/debug" #if os.path.exits(debug_file): enter ipdb
    result_file = "result.csv"




