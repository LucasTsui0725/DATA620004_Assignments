任务相关数据网盘链接：链接：https://pan.baidu.com/s/1_D_U3D7Ti-U0aiJKvtqtvg 提取码：0624 

含完整输出的模型链接：链接：https://pan.baidu.com/s/1FlDigntz1Kx2oEC_Vw2cuA 提取码：0624 



## 数据准备

### 物体扫描
我们首先要进行对物体进行扫描，扫描所采用的设备是：Redmi K60（输出视频像素为1260\*720）。进一步，我们自视频中每隔15帧抽取一张图片，最终共抽取121张图片。同时，考虑到之后训练所使用的显存为12G且建模的主要部分为小玩偶，我们截取了图像中心的640\*640的正方形区域作为最终的扫描结果，保存在jychai_data/raw_data下（实现详见cut_sampling.ipynb）。

### LLFF格式转化
该部分内容主要有两部分组成，先使用COLMAP获取相机位姿，再通过脚本文件将位姿数据转化为LLFF格式。

第一部分使用COLMAP获取的相机位姿。其结果如下。

![COLMAP_yibu](Repo_pic/COLMAP_yibu.PNG)

注：更详细的结果见视频jychai_data/COLMAP_yibu.mp4。

我们可以发现此时已能够观察到玩偶的部分轮廓。

第二部分中，我们需要将上面得到的数据转化为NeRF与DSNeRF所需要的LLFF格式，我们使用了与DSNeRF项目相同的imgs2poses.py进行实现。这一操作将会在yibu文件夹下添加一个poses_bounds.npy文件。我们将以上两部分的结果保存在jychai_data/yibu下。

## 模型简介

**balabala**

在本实验中，我们主要基于DSNeRF以及原始NeRF两个模型，并对其代码稍做了一些更改，保证其能够适配我们的数据在12G显存的RTX 3060上运行。两个项目的原始项目链接如下：
- DSNeRF：https://github.com/dunbar12138/DSNeRF
- NeRF：https://github.com/yenchenlin/nerf-pytorch/tree/master

### 实验设置
本实验的大部分设置均与DSNeRF、NeRF项目中的DEMOS的设置相同，主要的不同点如下：
1. 本实验的数据是360度的环绕数据，在训练时需要加入（*这个你看看怎么变成那种代码的样子* --spherify --no_ndc）
2. 为了减少计算开支，在配置文件我们令factor=8，即对图片进行八倍下采样。
3. 此外，我们还对其余的一些参数进行了修改，如视频输出频率等。详见网盘链接中的jychai_data/yibu_DSNeRF.txt文件与jychai_data/yibu_NeRF.txt文件。

**注：为了项目的精简性，我们上传Github的文件中删去了部分模型的输出，如模型训练前中期的参数与重建视频，而只保留了最后一轮的结果。如需完整的项目文件，请通过百度网盘链接下载。**

## 结果展示
两个模型的部分结果展示如下：
1. 输入示意图
![Model_input](Repo_pic/Model_input.PNG)
   
2. 输出3D重建模型三视图
![DSNerf_out](Repo_pic/DENeRF_out_3D.PNG)
![Nerf_out](Repo_pic/NeRF_out_3D.PNG)

完整的重建视频见文件jychai_data/yibu_DSNeRF.mp4与jychai_data/yibu_NeRF.mp4。

## 结果分析

从上面的结果中，我们可以发现DSNeRF与NeRF对于伊布小玩偶的重建效果相近，都能够勉强完成3D重建任务。但两者在训练时长上依然存在一定的差异，DSNeRF大约训练了3小时左右（50k个iteration）就取得了较好的结果，而NeRF为取得相近的结果大约需要7个小时左右，完整训练200k个iteration大约花了11小时。

此外，两个模型的性能还有可以提升的空间。一方面，我们目前受制于显卡的显存大小，迫不得已地将图片进行下采样，将640\*640的图片八倍下采样为了80\*80，在这一过程中显然损失了很多的信息。如果能够使用更强大的显卡，也许可以减小下采样的系数factor，保留更多的图像细节。另一方面，我们的扫描设备和扫描环境比较简陋，用了一个烟盒作为底座，背景也是若干张A4临时搭起来的，可能对模型不是特别友好，也许有更加好的扫描结果也可能增强模型性能。