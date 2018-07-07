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
1、CTC：用来实现端到端的语音识别，解决了文本序列与神经网络模型输出的一对多映射问题；<br>
2、1*1卷积核：最早见于2014年论文《Network In Network》，其作用是整合多通信信息进行非线性变换；<br>
3、残差网络：将之前的输入和跳跃若干层之后的输出相融合，Wavenet采用分block策略，前一个block的输入输出之和为下一个block的输入；<br>
4、空洞卷积（极黑卷积）：增大感受野，以提取较长距离文本上下文之间的关系，其效果如下图，第一幅图为传统的CNN模型，感受野为5，而第二幅图为dilated CNN模型，感受野为16，可见其可以在节约pooling层的情况下，通过更大的感受野获取更多的信息。<br>
5、Batch Normalization批标准化策略，见于论文《Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift》。<br>

实验
------
1、python chainer_train.py进行训练（文件路径需要修改）<br>
2、python test.py进行测试（文件路径需要修改）<br>

# Smart coatroom based on artificial intelligence technology

## Background
Clothing is the one of the most important part in our life as well as an important component of human civilization.<br>
In our daily life, everyone has many own clothes in the coatroom, and the majority of the men are lack of management over theirs clothes as well as the majority of women are willing to spend more time to match and change clothes before going outside. However, with the rapid development of technology, the more competitive pressure people have, the faster people live. Choosing clothes with our own eyes and senses every day and then match them quickly and rushed into the crowd will make our live too single and rough. At the same time, there are more and more smart home product integrated into people’s lives but artificial intelligence has few applications in the field of smart coatroom.<br>
Based on this background, the project is going to develop a coatroom to meet the public demand which used the artificial intelligence. The project is able to manage clothing automatically, interact with users by using some artificial intelligence technologies includes speech recognition, speech synthesis and multi-round dialog system in specific scene, and make some clothing matching recommendation for user through the recommendation algorithm.<br>
## My Work
There are four members in our team, and the main work of this project can be divided into specific scenarios clothing identification and classification, speech recognition, multi-round dialog system, speech synthesis, recommendation algorithm, and so on. My job is to train the model of speech recognition.
## Chainer
Chainer is a deep learning framework which is similar to Tensorflow and keras. The reason why I trained speech recognition model by using Chainer is that the source target of this project is to participate into the 11th Intel cup series, and the game party required us to use this framework.
## Speech Recognition Model
Based on my investigation, I found Google has proposed WAVENET model for speech recognition in “WAVENET: A GENERATIVE MODEL FOR RAW AUDIO”, and they also declare that this model can be used in speech recognition with great performance.<br>
There are some differences between WAVENET and traditional speech recognition model:<br>
1\ Using CTC loss to implement end-to-end speech recognition so that can avoid the overhead and impart of result alignment.<br>
2\ Using 1*1 convolution kernel to integrate multi-channel information and perform no-linear translation.<br>
3\ Using residual network to merge the previous input with the output after several layers so that the model can capture the relationship between the before and after time series in the speech.<br>
4\ Using dilated convolution kernel to increase the feeling field so that the model can extract the relationship between the longer distance text context.<br>
5\ Using batch normalization strategy ensures that the fusion objects have the same scale of evaluation indicators.<br>



