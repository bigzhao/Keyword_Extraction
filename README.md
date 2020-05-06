# “神策杯”2018高校算法大师赛第二名代码
> 队伍：发SCI才能毕业   

## 比赛信息
比赛链接：http://www.dcjingsai.com/common/cmpt/“神策杯”2018高校算法大师赛_竞赛信息.html

数据集issues里面有百度网盘链接

任务：训练出一个”关键词提取”的模型，提取10万篇资讯文章的关键词。

数据：1） all\_docs.txt，108295篇资讯文章数据，数据格式为：ID 文章标题 文章正文，中间由\\001分割。2） train\_docs_keywords.txt，1000篇文章的关键词标注结果，数据格式为：ID 关键词列表，中间由\\t分割。 

## 目录说明
- jieba：修改过的jieba库。
- 字典：存放jieba词库。PS：词库来源于搜狗百度输入法词库、爬虫获取的明星词条和LSTM命名实体识别结果。
- all_docs.txt: 训练语料库
- train_docs_keywords.txt：我把明显错误的一些关键词改回来了，例如D039180梁静茹->贾静雯、D011909泰荣君->泰容君等
- classes_doc2vec.npy：gensim默认参数的doc2vec+Kmeans对语料库的聚类结果。
- my_idf.txt：计算得来的语料库的idf文件。
- lgb_sub_9524764012949717.npy LGB的某一次预测值，用于特征生成
- stopword.txt：停用词
- Get_Feature.ipynb：特征生成notebook，对训练集和测试集生成对应的文件
- lgb_predict.py：预测并输出结果的脚本。需要train_df_v7.csv和test_df_v7.csv。
- train_df_v7.csv，test_df_v7.csv：Get_Feature.ipynb 跑出来的结果，notebook有详细特征说明
- word2vec模型下载地址：https://pan.baidu.com/s/1krH0ThIqvldmF5gfOZ6s7A 提取码：tw0m。
- doc2vec模型下载地址：链接：https://pan.baidu.com/s/17ZYAbTeqsXXq-hE3z3QqmA 提取码：0ciw.

## 运行说明
1. 运行Get_Feature.ipynb获取train_df_v7.csv和test_df_v7.csv.
2. 运行lgb_predict.py 获取结果sub.csv。


## 依赖包
```
numpy 1.14.0rc1
pandas 0.23.0
sklearn 0.19.0
lightgbm 2.0.5
scipy 1.0.0
```

## 解题思路方案说明
1. 利用jieba的tfidf方法筛选出Top20的候选关键词
2. 针对每条样本的候选关键词提取相应的特征，把关键词提取当作是普通二分类问题。特征可以分为以下两类：1）样本文档自身特征：例如文本的长度、句子数、聚类结果等；2）候选关键词自身特征：关键词的长度、逆词频等；3）样本文本和候选关键词的交互特征：词频、头词频、tfidf、主题相似度等；4）候选关键词之间的特征：主要是关键词之间的相似度特征。5）候选关键词与其他样本文档的交互特征：这里有两个非常强的特征，第一是在整个数据集里被当成候选关键词的频率，第二个与点击率类似，算在整个文档中预测为正样本的概率结果大于0.5的数量（在提这个特征的时候我大概率以为会过拟合，但是效果出乎意料的好，所以也没有做相应的平滑，或许是因为结果只选Top2的关键词，这里概率选0.5会有一定的平滑效果，具体操作请看lgb_predict.py的31-42行）。
3. 利用LightGBM解决上述二分类问题，然后根据LightGBM的结果为每条文本选出预测概率Top2的词作为关键词输出即可。

