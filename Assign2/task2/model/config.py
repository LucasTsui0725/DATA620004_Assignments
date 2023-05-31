import torch
BATCH_SIZE = 8
RESIZE_TO = 512
NUM_EPOCHS = 40
DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

TRAIN_DIR = 'G:/BaiduNetdiskWorkspace/working_place/NN_DL/fast_rcnn_fcos/data/dataset_jychai/train'

VALID_DIR = 'G:/BaiduNetdiskWorkspace/working_place/NN_DL/fast_rcnn_fcos/data/dataset_jychai/val'

CLASSES = ['aeroplane',
           'bicycle',
           'bird',
           'boat',
           'bottle',
           'bus',
           'car',
           'cat',
           'chair',
           'cow',
           'diningtable',
           'dog',
           'horse',
           'motorbike',
           'person',
           'pottedplant',
           'sheep',
           'sofa',
           'train',
           'tvmonitor']
NUM_CLASSES = 20

VISUALIZE_TRANSFORMED_IMAGES = False

OUT_DIR_model = 'G:/BaiduNetdiskWorkspace/working_place/NN_DL/fast_rcnn_fcos/result/model'

OUT_DIR_TF_Board = 'G:/BaiduNetdiskWorkspace/working_place/NN_DL/fast_rcnn_fcos/result/board'

OUT_DIR_PIC = 'G:/BaiduNetdiskWorkspace/working_place/NN_DL/fast_rcnn_fcos/result/pic'