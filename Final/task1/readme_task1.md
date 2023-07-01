## 文件介绍

1. ./log文件夹。主要用于保存训练过程中的若干结果，如训练中的loss曲线、训练集上的Acc等。可以通过tensorboard查看。
2. ./pic文件夹。主要用于保存模型的可视化结果（撰写报告使用）。
3. ./Code文件夹：共包含5个文件。
   1. baseline_resnet18.py，Pretrain_resnet18.py文件。主要用于实现Resnet18主体，以及多个下游任务头（如：rotation头、分类头等）
   2. 若干jupyter文件。主要用于实现模型的训练以及测试部分。

当前任务数据集链接：链接：https://pan.baidu.com/s/15n4WqAej7GlB_SR4SgWvOA 提取码：0624 
当前任务模型参数链接：链接：https://pan.baidu.com/s/1536IUQP_BCP53yedmwOiWQ 提取码：0624 

## 训练步骤

1. baseline model

   直接运行resnet18_without_pre.ipynb文件，将会在log文件夹下保存训练过程中的loss曲线、Acc曲线、F1曲线以及在测试集上的Acc曲线，并在最后将模型保存到model_path文件下（resnet18.pt是resnet18主体，resnet18_clf_head.pt是分类头）
2. Pretrain model

   1.预训练步骤

   运行resnet18_with_pre.ipynb文件下的Pretrain step部分，该部分代码将会分别进行Color以及Rotation的预训练。
   当完成Rotation的预训练后，将会在log文件夹下保存训练过程中的loss曲线、Acc曲线、F1曲线以及在测试集上的Acc曲线，并每5个Epoch保存一次Resnet主体模型。
   当完成Color的预训练后，将会在log文件夹下保存训练过程中以及在测试集上的Loss（MSE）曲线，并每5个Epoch保存一次Resnet主体模型。

   2.Linear Classification Protocol 步骤

   运行resnet18_with_pre.ipynb文件下的Linear Classification Protocol step部分，该部分代码将会进行Linear Classification Protocol。该部分代码将在model_path下保存针对pretrain_resnet18_color_24.pt和pretrain_resnet18_rotation_24.pt的分类头（resnet18_color_clf_head.pt和resnet18_rotation_clf_head.pt）。

## 测试步骤

直接运行preformance_result.ipynb。将会给出三种不同情况下的训练结果。分别是无预训练的ResNet18、使用Color进行预训练的ResNet18和使用Rotation进行预训练的ResNet18在测试集上的Acc结果。
