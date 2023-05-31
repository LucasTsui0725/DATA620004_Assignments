## Task1

### 训练步骤

1. 首先保证数据已经下载，可以使用数据存储路径或直接运行**main.sh**脚本来下载数据到dataset文件夹
2. 运行**main.sh**文件来进行模型的训练

   ```
   # 原始模型
   bash mian.sh --input-channel 3 --output-channel 100 -v --weight-decay <weight_decay> --gpu <gpu> --metrics accuracy precision recall f1_score --seed <seed> 
   # 数据增强模型
   bash main.sh --input-channel 3 --output-channel 100 -v --augment <augment_method> --prob <prob> --beta <beta> --weight-decay <weight_decay> --gpu <gpu> --metrics accuracy precision recall f1_score --seed <seed> 
   ```

### 测试步骤

1. 对于数据增强模型下的训练数据情况可视化，参照**show_img.ipynb**
2. 对于训练好的模型进行结果验证，参照**generate_metric.ipynb**
