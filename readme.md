SpeechChainer——基于chainer实现中文语音识别
====

背景
------
2016年Google发布了Wavenet语音合成系统，而在论文《WAVENET: A GENERATIVE MODEL FOR RAW AUDIO》中介绍了Wavenet网络可用来解决的问题，其中包括了语音识别。<br>

基本结构
------
Wavenet系统架构融合了dilated CNN、残差网络、CTC、LSTM中的门以及1*1卷积核等经典结构，具体可见于目录下文件SpeechChainer.pdf。<br>

数据集
------
1、[清华30小时中文数据集](http://www.openslr.org/18/)<br>
2、[AISHELL178小时中文数据集](http://www.aishelltech.com/newsitem/277940177)<br>

基本技术
------
1. CTC：用来实现端到端的语音识别，解决了文本序列与神经网络模型输出的一对多映射问题；<br>
2. 1*1卷积核：最早见于2014年论文《Network In Network》，其作用是整合多通信信息进行非线性变换；<br>
3. 残差网络：将之前的输入和跳跃若干层之后的输出相融合，Wavenet采用分block策略，前一个block的输入输出之和为下一个block的输入；<br>
4. 空洞卷积（极黑卷积）：增大感受野，以提取较长距离文本上下文之间的关系，其效果如下图，第一幅图为传统的CNN模型，感受野为5，而第二幅图为dilated CNN模型，感受野为16，可见其可以在节约pooling层的情况下，通过更大的感受野获取更多的信息。<br>
5. Batch Normalization批标准化策略，见于论文《Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift》。<br>
