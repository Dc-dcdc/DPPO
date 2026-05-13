# DPPO
先用扩散模型预训练并使用强化学习PPO算法进行微调
## ✨ Pretrain部分 
1. 训练代码位于 `pretrain/train_pretrain.py`，训练对应的配置参数位于`configs/pretrain/policy`，训练不同的任务时，需要注意修改`train_pretrain.py`中的`env`(决定场景)和`policy`(决定训练策略：ACT、diffusion)。
2. 模型快照和评估视频储存的默认位置位于`configs/pretrain/pre_default.yaml`中的hydra.run.dir,这是相对于该项目（DPPO）的相对保存位置，注意命令行的运行位置，否则会存到其他地方，wandb的保存文件名为hydra.run.job。
3. 预训练代码设置了断点续训，输入保存的模型快照路径并设置`resume=true`即可，会自动读取训练时使用的policy配置参数。
4. 评估代码位于`pretrain/eval.py`，输入模型快照的路径即可，会自动读取训练时使用的policy配置参数，可以在`eval.py`设置`render_camera=['overhead_cam']`来设置录制视频的视角。
5. 值得注意的是，这里用到了av-aloha的lerobot代码，换设备训练需要注意，后期可以注意更新为官网版本的lerobot
## ✨ Finetune部分 
1. 微调代码位于`finetune/train_finetune.py`，输入模型快照的路径即可，会自动读取训练时使用的policy配置参数
2. 为了适配评估代码`pretrain/eval.py`，保存权重的同时生成了对应的训练参数配置表`config.yaml`和`config.json`


## ✨ sim_env部分 
1. 添加了模型推理部分，可以添加训练好的模型进行在线仿真推理，可以修改display_cameras参数获取要单独渲染的相机视角，一行两个进行排布。此外还可以通过修改代码中SIM_DT为具体值，从而实现慢速的观测效果。


## 🧾 小贴士
1. mujoco环境中`aloha_real.xml`比`aloha_sim.xml`多出以下两处聚光灯：
```
<light mode="targetbodycom" target="left_gripper_link" pos="-.5 .7 2.5" cutoff="55"/>
<light mode="targetbodycom" target="right_gripper_link" pos=".5 .7 2.5" cutoff="55"/>
```
2. 且两者的双目相机广角不一样，`aloha_real.xml`中`fovy="90"`，而`aloha_sim.xml`中为`fovy="66.21"`
```
<camera name="zed_cam_left" pos="0.03 0.00119254 -0.04325" euler="1.57079632679 0 3.14159265359" fovy="66.21" mode="fixed"/>
<camera name="zed_cam_right" pos="-0.03 0.00119254 -0.04325" euler="1.57079632679 0 3.14159265359" fovy="66.21" mode="fixed"/> 
```
3. 可以在评估eval.py代码中查看推理时间，一把来说一次会推理horizon步，这一次推理时间是最久的，后续只从推理的动作中取出并执行即可，所以推理时间会呈现 类似[31.86 ms、0.26 ms、0.31 ms、0.29 ms...]的分布,长度是实际执行的步数n_action_steps