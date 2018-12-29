CrossDomainReviewTextRecommendation
===================================

基于评论文本的跨领域商品推荐
----------------------------
关注于评分预测问题

* 利用用户对商品的评论文本进行建模, 获取用户向量以及商品向量表示
* 利用DSN [#]_ 等Domain Adaptation方法构建用户在源领域与目标领域上的关联
* 利用待推荐用户在目标领域上的向量为用户在源领域上进行推荐

所有的实验基于Amazon公开的数据集
下载地址: http://jmcauley.ucsd.edu/data/amazon/

Good Luck

目录说明
--------
This is a paragraph. Its' quite
short.

::


  ├── CDRTR
  │   ├── core
  │   │   ├── DeepModel: DL代码路径, 包括SentiRec, DSN, DSNRec
  │   │   └── LinearModel: 包括LR
  │   ├── dataset: 为SentiRec, DSNRec 编写的数据集类
  │   ├── preprocess: 文本预处理工具, 包括, 两个领域词典生成,
                      词向量化, 统计领域用户交叠情况, 冷用户统计等
  ├── exam: 实验代码路径
  │   ├── data: 样例数据集
  │   │   ├── preprocess: 预处理数据输出路径
  │   │   └── source: 原数据存放路径
  │   ├── log
  │   │   ├── baseline
  │   │   ├── debug_generateVoca_MultiCross
  │   │   ├── debug_mergeUI_MultiCross
  │   │   ├── DSNRec
  │   │   └── sentitrain
  │   ├── MultiCross: 多领域相互跨领域推荐实验数据路径
  │       ├── Beauty_Clothin_Shoes_and_Jewelry
  │       ├── Beauty_Movies_and_TV
  │       ├── Kindle_Beauty
  │       ├── Kindle_Clothin_Shoes_and_Jewelry
  │       ├── Kindle_Movies_and_TV
  │       └── Movies_and_TV_Clothin_Shoes_and_Jewelry
  ├── tests: 测试代码路径

实验运行方法
---------------
在exam目录下面有一个makefile, 内容如下:

::

    export PYTHONPATH=..

    transCSV:
        python -m CDRTR.preprocess csv_format --mode $(MODE) --fields reviewerID,asin,overall --data_dir $(DATA)

    # make generatevoca MODE=DEBUG DATA=data
    generatevoca:
        python -m CDRTR.preprocess generatevoca --mode $(MODE) --fields asin,reviewerID,overall,reviewText,unixReviewTime --data_dir $(DATA)

    # make extractinfo MODE=DEBUG
    extractinfo:
        python -m CDRTR.preprocess extractinfo --mode $(MODE)

    preprocess: generatevoca extractinfo

    # make sentitrain DATA=data DOMAIN=Music
    # make sentitrain DATA=data DOMAIN=Auto
    sentitrain:
        python -m CDRTR.core.DeepModel.SentiRec --dir $(DATA)/preprocess --domain $(DOMAIN) --filter_size 3,5,7,11 --epoches 400

    mergeUI:
        python -m CDRTR.preprocess mergeui --mode $(MODE) --data_dir $(DATA)

    DSNRec:
        python -m CDRTR.core.DeepModel.DSNRec --data_dir $(DATA) --src_domain $(SRCDO) --tgt_domain $(TGTDO) --epoches $(EPOCH) --mode $(MODE)

makefile中设置了多个任务, 包括:

1. generatevoca: 预处理原json数据(存放在data/source/), 该任务将统计两个领域数据, 生成词典, 冷用户情况等;

2. transCSV: 将json格式的原数据处理为csv格式, 方便使用MyMediaLite进行baseline实验

3. sentitrain: 对两个领域用户评论评分数据进行sentiRec [#]_ 训练, 获取CNN层输出作为评论句子的向量表示

4. mergeUI: 将sentitrain输出的评论文本聚合为用户, item的向量表示

5. DSNRec: 对两个领域数据进行跨领域推荐

exam目录下的pipeline.sh脚本文件, 包含了上述几个步骤的运行

::

    Usage pipeline.sh data_dir src_domain tgt_domain epoch

example:

::

  winchua@CCNLForDL:~/CrossDomainReviewTextRecommendation/CDRTR/exam$ ./pipeline.sh data Auto Musi 400
  ...
  ...
  # 运行了多个实验之后, 可以通过如下脚本查看各个领域针对完全冷启动用户的评分预测RMSE结果
  winchua@CCNLForDL:~/CrossDomainReviewTextRecommendation/CDRTR/exam$ python show_result.py `find . -name "test*.pk" | grep -v MultiCross`
  +-------------+----------------+
  |    domain   |      rmse      |
  +-------------+----------------+
  |     Cell    | 1.02770589547  |
  |   Digital   |  0.9646660533  |
  | Electronics | 1.09582470097  |
  |    Kindle   | 1.01310818718  |
  |     Musi    | 0.737983482096 |
  |    Tools    | 0.94334647007  |
  |     CDs     | 1.00803053039  |
  |    Video    | 0.987097572114 |
  |   Jewelry   | 1.04308169038  |
  |    Movie    | 1.00178416516  |
  |    Beauty   | 1.09342681075  |
  |    Sports   | 0.901738399564 |
  |     Auto    | 0.937711151504 |
  |    Office   | 0.855115969443 |
  |     Toys    | 0.911580666733 |
  +-------------+----------------+

  winchua@CCNLForDL:~/CrossDomainReviewTextRecommendation/CDRTR/exam$ python show_user_record_count.py `ls -F | grep / | grep -v log | grep -v MultiCross | cut -d/ -f1`
  +----------------------------------------------+--------------+--------------+--------------+------------------+-----------------+
  |                   domains                    | srcUserCount | tgtUserCount | overlapCount |  srcOverlapRate  |  tgtOverlapRate |
  +----------------------------------------------+--------------+--------------+--------------+------------------+-----------------+
  |      Beauty/Clothing_Shoes_and_Jewelry       |    18143     |    35167     |     4220     |  0.232596593728  |  0.11999886257  |
  |          CDs_and_Vinyl/Electronics           |    68998     |    186143    |     6260     |  0.090727267457  | 0.0336300586109 |
  |   Cell_Phones_and_Accessories/Video_Games    |    26480     |    22904     |     1399     |  0.052832326284  | 0.0610810338805 |
  |         Toys_and_Games/Digital_Music         |    19233     |     5362     |     179      | 0.00930692039723 | 0.0333830660201 |
  | Sports_and_Outdoors/Grocery_and_Gourmet_Food |    33563     |    12646     |     2035     | 0.0606322438399  |  0.160920449154 |
  |          Kindle_Store/Movies_and_TV          |    65469     |    121206    |     2754     | 0.0420657104889  | 0.0227216474432 |
  |  Office_Products/Tools_and_Home_Improvement  |     3012     |    14745     |     1893     |  0.628486055777  |  0.128382502543 |
  +----------------------------------------------+--------------+--------------+--------------+------------------+-----------------+

  +----------------------------------------------+------------+---------------+------------+---------------+
  |                   domains                    | srcURcount | srcOverRcount | tgtURcount | tgtOverRcount |
  +----------------------------------------------+------------+---------------+------------+---------------+
  |      Beauty/Clothing_Shoes_and_Jewelry       |   149091   |     49411     |   238420   |     40257     |
  |          CDs_and_Vinyl/Electronics           |   968881   |     128711    |  1589951   |     99237     |
  |   Cell_Phones_and_Accessories/Video_Games    |   179952   |     14487     |   211931   |     19849     |
  |        Automotive/Musical_Instruments        |   20117    |      356      |    9842    |      419      |
  |         Toys_and_Games/Digital_Music         |   164876   |      2721     |   61975    |      2731     |
  | Sports_and_Outdoors/Grocery_and_Gourmet_Food |   273513   |     22824     |   114343   |     36911     |
  |          Kindle_Store/Movies_and_TV          |   932521   |     50098     |  1628754   |     68779     |
  |  Office_Products/Tools_and_Home_Improvement  |   26170    |     27088     |   111009   |     23467     |
  +----------------------------------------------+------------+---------------+------------+---------------+
  winchua@CCNLForDL:~/CrossDomainReviewTextRecommendation/CDRTR/exam$ python baseline_result_show.py
  +----------------------------+-------------------------------+---------------------------+----------+-------------+---------+-----------------+---------------+-------------+-----------------------------+
  |           domain           | FactorWiseMatrixFactorization | BiasedMatrixFactorization | SlopeOne | SVDPlusPlus |  Random | BiPolarSlopeOne | GlobalAverage | ItemAverage | LatentFeatureLogLinearModel |
  +----------------------------+-------------------------------+---------------------------+----------+-------------+---------+-----------------+---------------+-------------+-----------------------------+
  |            Musi            |            0.74917            |          0.75819          | 0.74003  |    0.7429   | 2.07642 |     0.74003     |    0.74003    |   0.79938   |            2.5739           |
  |       Toys_and_Games       |            0.91575            |          0.90545          | 1.01175  |   0.94514   | 1.96148 |     1.01175     |    1.01175    |   0.91091   |           2.22349           |
  | Tools_and_Home_Improvement |            0.92429            |          0.93079          | 0.94901  |   0.93271   | 2.02191 |     0.94901     |    0.94901    |   0.98844   |           3.01554           |
  |          Jewelry           |            1.07836            |          1.08417          | 1.09712  |   1.08301   | 2.02008 |     1.09712     |    1.09712    |   1.12147   |           2.63191           |
  |           Beauty           |             1.1154            |          1.12241          | 1.14271  |   1.11946   | 2.01473 |     1.14271     |    1.14271    |   1.16054   |           2.46513           |
  |      Office_Products       |            0.86723            |          0.86452          |   0.91   |   0.88177   | 1.99483 |       0.91      |      0.91     |   0.89912   |           2.39326           |
  |            Auto            |            0.91939            |          0.91734          | 0.95858  |   0.93183   | 2.06752 |     0.95858     |    0.95858    |   0.95209   |           2.65961           |
  |       CDs_and_Vinyl        |            0.95356            |          0.95376          | 1.00893  |   0.96628   | 2.02005 |     1.00893     |    1.00893    |   0.96833   |           2.60192           |
  |        Cell_Phones         |            1.04561            |          1.05164          | 1.09231  |   1.05525   | 1.98757 |     1.09231     |    1.09231    |   1.06846   |           2.19888           |
  |          Grocery           |            0.98135            |          0.98238          |  1.0676  |   0.99329   | 1.94582 |      1.0676     |     1.0676    |   0.99631   |           2.15443           |
  |           Sports           |            0.90509            |          0.90759          | 0.93544  |    0.9153   |  2.0045 |     0.93544     |    0.93544    |   0.94513   |           2.72289           |
  |        Video_Games         |            1.00654            |          1.00706          | 1.08674  |   1.02059   | 1.96103 |     1.08674     |    1.08674    |   1.02006   |           1.98473           |
  |       Digital_Music        |            0.99705            |          0.98776          | 1.08185  |   1.01934   | 1.97827 |     1.08185     |    1.08185    |   0.99791   |           2.34118           |
  |           Kindle           |            0.99793            |           0.9961          | 1.04402  |   1.01122   | 1.96238 |     1.04402     |    1.04402    |    1.0107   |           2.61299           |
  |        Electronics         |            1.07022            |          1.07283          | 1.12268  |   1.08046   | 2.01596 |     1.12268     |    1.12268    |   1.08378   |           1.91997           |
  |       Movies_and_TV        |            0.97773            |          0.96989          | 1.09381  |   0.99628   | 1.96544 |     1.09381     |    1.09381    |   0.96867   |           1.73305           |
  +----------------------------+-------------------------------+---------------------------+----------+-------------+---------+-----------------+---------------+-------------+-----------------------------+


Tensorboard查看模型结构
-----------------------

::

  winchua@CCNLForDL:~/CrossDomainReviewTextRecommendation/CDRTR/exam$ cd log/DSNRec/Auto_Musi/
  winchua@CCNLForDL:~/CrossDomainReviewTextRecommendation/CDRTR/exam/log/DSNRec/Auto_Musi$ ls
  test_mses.pk  train  Untitled.ipynb
  winchua@CCNLForDL:~/CrossDomainReviewTextRecommendation/CDRTR/exam/log/DSNRec/Auto_Musi$ tensorboard --logdir=train --port 12345


.. image:: https://raw.githubusercontent.com/WinChua/CDRTR/master/docs/source/_static/model.bmp

参考引用
--------

.. [#] Bousmalis, K., Trigeorgis, G., Silberman, N., Krishnan, D., & Erhan, D. (2016). Domain SeparationNetworks, (Nips). Retrieved from http://arxiv.org/abs/1608.06019
.. [#] Hyun, D., Park, C., Yang, M.-C., Song, I., Lee, J.-T., & Yu, H. (2018). Review Sentiment-Guided Scalable DeepRecom-mender System. Ann SIGIR, 18, 965–968. https://doi.org/10.1145/3209978.3210111



✨🍰✨
