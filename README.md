## LearnDeepLearning
使用keras实现经典的神经网络结构

## 文件布局
* Log/ 学习过程中的笔记  
* src/ 实现源码  
	* dpl/ 辅助接口
  	* traditional/ - 经典网络实现  
  		* base/ - 多层感知机等基础神经网络
    	* cnn/ - 卷积网络（包括alex_net google_net le_net res_net shallow_net vgg16_net zf_net）  
    	*  segment/ - 分割网络(包括 fcn 8s 16s 32s u_net)  
    	*  setup_nn.ipynb - 基础神经网络搭建示例（全连接32 64 128 256 512 1024 全连接+卷积++maxpool+Dropout等组合） 
    	*  setup_cnn.ipynb  - cnn网络搭建示例  
    	*  setup_rnn.ipynb  - rnn网络搭建示例  
    	*  setup_segnet.ipynb  - sgment网络搭建示例 
  
## 依赖 ##
  [The Jupter NoteBook](https://jupyter.org/) ([安装指南](https://jupyter.readthedocs.io/en/latest/install.html))  
  

