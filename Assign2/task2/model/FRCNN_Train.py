from utils import get_mAP, get_mIOU
from config import DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR_model, OUT_DIR_TF_Board
from config import VISUALIZE_TRANSFORMED_IMAGES

from torch.utils.tensorboard import SummaryWriter
from sympy import im

from Faster_RCNN import create_model_FRCNN

from tqdm.auto import tqdm
from dataset_jychai import train_loader, valid_loader
import torch
import torch.utils
import matplotlib.pyplot as plt
import time
plt.style.use('ggplot')


# function for running training iterations


def train(train_data_loader, model):
    print('Training')
    global train_itr
    global train_loss_list
    global writer

    # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))

    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)
        writer_tfb.add_scalar('loss_train', loss_value, len(train_loss_list))
        losses.backward()
        optimizer.step()
        train_itr += 1

        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return train_loss_list

# function for running validation iterations


def validate(valid_data_loader, model):
    print('Validating')
    global val_itr
    global val_loss_list
    global writer_tfb

    # initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))

    mAP_list = []
    mIOU_list = []
    acc_list = []

    for i, data in enumerate(prog_bar):
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            model.eval()
            detections = model(images)

            for i in range(len(detections)):
                target_class = targets[i]['labels'].cpu().numpy().tolist()
                target_boxes = targets[i]['boxes'].cpu().numpy().tolist()

                pre_scores = detections[i]['scores'].cpu().detach().numpy()
                pre_boxes = detections[i]['boxes'].cpu(
                ).detach().numpy().tolist()
                pre_class = detections[i]['labels'].cpu(
                ).detach().numpy().tolist()
                mAP_tmp, acc_tmp = get_mAP(
                    pre_class, pre_boxes, pre_scores, target_class, target_boxes, iou_threshold=0.5)
                mAP_list.append(mAP_tmp)
                acc_list.append(acc_tmp)
                mIOU_list.append(
                    get_mIOU(pre_boxes, target_boxes, iou_threshold=0.5))
            model.train()
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        writer_tfb.add_scalar('loss_val', loss_value, len(val_loss_list))
        writer_tfb.add_scalar('mAP_val', sum(mAP_list)/len(mAP_list), len(val_loss_list))
        writer_tfb.add_scalar('mIoU_val', sum(mIOU_list)/len(mIOU_list), len(val_loss_list))
        writer_tfb.add_scalar('Acc_val', sum(acc_list)/len(acc_list), len(val_loss_list))

        val_itr += 1
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return val_loss_list


if __name__ == '__main__':
    writer_tfb = SummaryWriter(OUT_DIR_TF_Board+'/log_FRCNN')
    # initialize the model and move to the computation device
    model = create_model_FRCNN(num_classes=NUM_CLASSES)

    model = model.to(DEVICE)
    # get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    # define the optimizer
    optimizer = torch.optim.SGD(
        params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    train_itr = 1
    val_itr = 1
    # train and validation loss lists to store loss values of all...
    # ... iterations till ena and plot graphs for all iterations
    train_loss_list = []
    val_loss_list = []
    # name to save the trained model with
    MODEL_NAME = 'model'
    # whether to show transformed images from data loader or not
    if VISUALIZE_TRANSFORMED_IMAGES:
        from utils import show_tranformed_image
        show_tranformed_image(train_loader)
    # start the training epochs
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")
        # reset the training and validation loss histories for the current epoch

        # start timer and carry out training and validation
        start = time.time()
        train_loss = train(train_loader, model)
        val_loss = validate(valid_loader, model)

        if (epoch+1) % 10 == 0:  # save model after every n epochs
            torch.save(model.state_dict(),
                       f"{OUT_DIR_model}/FRCNN_model{epoch+1}.pth")
            print('SAVING MODEL COMPLETE...\n')

