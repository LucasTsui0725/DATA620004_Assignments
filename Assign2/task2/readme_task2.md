## Task2

### 训练步骤

1. 将数据保存在dataset_jychai下，分为三个文件夹：train，val，test。分别保存了训练集、验证集以及测试集的文件。
   * 在训练集与验证集中，保存了图片文件（jpg文件）以及对应的标注文件（xml文件）。具体的模型构建过程可以参考model/dataset_split.ipynb文件。
   * 在测试集中保存了我自己拍摄的三张小猫咪的照片。
2. **Faster-RCNN的训练**：运行FRCNN_Train.py文件。模型的训练过程通过tensorboard保存在result/board/log_FRCNN中；训练好的模型保存在result/model下，格式为FRCNN_modelXX.pth，其中XX代表了保存模型时的epoch轮数。
3. **FCOS的训练**：运行FCOSTrain.py文件。模型的训练过程通过tensorboard保存在result/board/log_FCOS中；训练好的模型保存在result/model下，格式为FCOS_modelXX.pth，其中XX代表了保存模型时的epoch轮数。
4. 我也通过截图的方式在result/pic/tensorboard_pic下保存了通过tensorboard所保存模型Train和Val结果。

### 测试步骤

1. **Faster-RCNN的测试**：运行FRCNN_Inference.py文件。
   模型会读取训练得到的模型FRCNN_model40.pth。
   * 若设置第18行Proposal_box = True，则模型会在Val集合中随机选择4张图片，并输出一阶段的proposal box的结果，保存在result/pic/FRCNN_pro_box中。
   * 若设置第18行Proposal_box = False，则模型会输出在测试集上的图像检测结果，保存在result/pic/FRCNN_test中。
2. **FCOS的测试**：运行FCOS_Inference.py文件。
   模型会读取训练得到的FCOS_model40.pth。模型会输出在测试集上的图像检测结果，保存在result/pic/FCOS_test中。

### 其他注意事项

1. 由于模型和数据集都太大了，故没有上传至github，保存到了百度网盘中。
   1. 数据集链接：链接：https://pan.baidu.com/s/1zWzlri0Fr04mI9EckmvOTg 提取码：0531
   2. 模型链接（result/model，result中还包括了其余的输出结果）：链接：https://pan.baidu.com/s/1x6Zudjdw_zwrNjKVpidhxQ 提取码：0531
