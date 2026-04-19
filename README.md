# DPPO
先用扩散模型预训练并使用强化学习PPO算法进行微调
## ✨ Pretrain部分 
1. 训练代码位于 `pretrain/train_diffusion.py`，训练对应的配置参数位于`configs/pretrain/policy`，训练不同的任务时，需要注意修改`train_diffusion.py`中的`env`(决定场景)和`policy`(决定训练策略：ACT、diffusion)。
2. 模型快照和评估视频储存的默认位置位于`configs/pretrain/pre_default.yaml`中的hydra.run.dir,这是相对于该项目（DPPO）的相对保存位置，注意命令行的运行位置，否则会存到其他地方，wandb的保存文件名为hydra.run.job。
3. 预训练代码设置了断点续训，输入保存的模型快照路径并设置`resume=true`即可，会自动读取训练时使用的policy配置参数。
4. 评估代码位于`pretrain/eval.py`，输入模型快照的路径即可，会自动读取训练时使用的policy配置参数，可以在`eval.py`设置`render_camera=['overhead_cam']`来设置录制视频的视角。
5. 值得注意的是，这里用到了av-aloha的lerobot代码，换设备训练需要注意，后期可以注意更新为官网版本的lerobot
## ✨ Finetune部分 
微调代码位于`finetune/train_finetune.py`，输入模型快照的路径即可，会自动读取训练时使用的policy配置参数