在进行训练前，**请务必确保已安装相关依赖！**可以参照DSNeRF项目和nerf-pytorch项目中的Readme文档进行配置。

DSNeRF模型训练方法：

- 下载数据链接中的jychai_data/yibu文件夹，并将其保存在DSNeRF/data下。
- 下载数据链接中的jychai_datai/yibu_DSNeRF.txt文件，并将其保存在DSNeRF/configs下。
- 终端输入 cd DSNeRF，切换至DSNeRF目录下。
- 终端输入 python run_nerf.py --config configs/yibu_DSNeRF.txt --spherify --no_ndc。默认将使用我们保存的模型参数进行测试，如希望从头训练，请删除logs/yibu文件夹下的tar文件。
- 耐心等待一段时间，DSNeRF的结果就会出现在DSNeRF/logs/release/yibu文件夹中。经验上来看大概需要**3~4**小时完成完整的训练。

NeRF模型训练方法：

- 下载数据链接中的jychai_datai/yibu文件夹，并将其保存在nerf-pytorch/data/nerf_llff_data下。
- 下载数据链接中的jychai_datai/yibu_NeRF.txt文件，并将其保存在nerf-pytorch/configs下。
- 终端输入 cd nerf-pytorch，切换至nerf-pytorch目录下。
- 终端输入 python run_nerf.py --config configs/yibu_NeRF.txt --spherify --no_ndc。默认将使用我们保存的模型参数进行测试，如希望从头训练，请删除logs/release/yibu文件夹下的tar文件。
- 耐心等待一段时间，NeRF的结果就会出现在nerf-pytorch/logs/yibu文件夹中。经验上来看大概需要**10~12**小时完成完整的训练。

**注：为了项目的精简性，我们上传Github的文件中删去了部分模型的输出，如模型训练前中期的参数与重建视频，而只保留了最后一轮的结果。如需含完整输出的项目文件，请通过百度网盘链接下载。**
