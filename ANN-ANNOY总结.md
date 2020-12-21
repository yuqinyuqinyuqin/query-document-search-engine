ANN-ANNOY总结

# 1.最近邻检索（Nearest Neighbor Search）



**最近邻检索**就是根据数据的相似性，从数据库中寻找与目标数据最相似的项目。这种相似性通常会被量化到空间上数据之间的距离，可以认为数据在空间中的距离越近，则数据之间的相似性越高。

**k最近邻（K-Nearest Neighbor，K-NN）检索**：当需要查找离目标数据最近的前k个数据项时。

**最近邻检索是线性复杂度**的，不能满足对于大规模数据检索的时间性能要求。

# 2.ANN（Approximate Nearest Neighbor）

面对庞大的数据量以及数据库中高维的数据信息，现有的基于 NN 的检索方法无法获得理想的检索效果与可接受的检索时间。因此，研究人员开始关注近似最近邻检索因此，最佳实践是使用**逼近方法搜索最近邻**。目前，近似最近邻缩短搜索时间有两类基本方案：第一类是缩短距离计算的时间，例如降维，第二类是减少距离计算的次数。

宏观上对ANN有下面的认知显得很有必要：**brute-force搜索的方式是在全空间进行搜索，为了加快查找的速度，几乎所有的ANN方法都是通过对全空间分割，将其分割成很多小的子空间，在搜索的时候，通过某种方式，快速锁定在某一（几）子空间，然后在该（几个）子空间里做遍历**。可以看到，正是因为缩减了遍历的空间大小范围，从而使得ANN能够处理大规模数据的索引。

 

主要分为：

树方法，如 KD-tree，Ball-tree，Annoy

哈希方法，如 Local Sensitive Hashing (LSH)

矢量量化方法，如 Product Quantization (PQ)

近邻图方法，如 Hierarchical Navigable Small World (HNSW)

#  3.树方法-annoy

## 3.1原理介绍

- annoy 算法的目标：建立一个数据结构能够在较短的时间内找到任何查询点的最近点，在精度允许的条件下通过牺牲准确率来换取比暴力搜索要快的多的搜索速度。

- 如下图如所示：一个二叉树来使得每个点查找时间复杂度：O(logn)。

### 建立索引

- 随机选择两个点，以这两个节点为初始中心节点，执行聚类数为2的kmeans过程，最终产生收敛后两个聚类中心点。这两个聚类中心点之间连一条线段（灰色短线），建立一条垂直于这条灰线，并且通过灰线中心点的线（黑色粗线）。这条黑色粗线把数据空间分成两部分。在多维空间的话，这条黑色粗线可以看成等距垂直超平面。在划分的子空间内进行不停的递归迭代继续划分，直到每个子空间最多只剩下K个数据节点。通过多次递归迭代划分的话，最终原始数据会形成二叉树结构。二叉树底层是叶子节点记录原始数据节点，其他中间节点记录的是分割超平面的信息。Annoy建立这样的二叉树结构是希望满足这样的一个假设:相似的数据节点应该在二叉树上位置更接近，一个分割超平面不应该把相似的数据节点分在二叉树的不同分支上。

### 查询过程

- 二叉树底层是叶子节点记录原始数据节点，其他中间节点记录的是分割超平面的信息。Annoy建立这样的二叉树结构是希望满足这样的一个假设:相似的数据节点应该在二叉树上位置更接近，一个分割超平面不应该把相似的数据节点分在二叉树的不同分支上。查找的过程就是不断看查询数据节点在分割超平面的哪一边。从二叉树索引结构来看，就是从根节点不停的往叶子节点遍历的过程。通过对二叉树每个中间节点（分割超平面相关信息）和查询数据节点进行相关计算来确定二叉树遍历过程是往这个中间节点左子节点走还是右子节点走。

![img](https://netease-we.feishu.cn/space/api/box/stream/download/asynccode/?code=9e030b26e810161f5e4315cc5057e82c_8f118824ce50c961_boxcnxZpX8hD6ggy57T3eWkxepc_66NrV4Fuwu5Z0EQITf2ZlE9cjfi8y4vq)

### **返回最终近邻节点**

- 每棵树都返回一堆近邻点后，如何得到最终的Top N相似集合呢？首先所有树返回近邻点都插入到优先队列中，求并集去重, 然后计算和查询点距离， 最终根据距离值从近距离到远距离排序， 返回Top N近邻节点集合。

## 3.2annoy最突出特性

- 使用**静态索引文件**，这意味着**不同进程可以共享索引**。

## 3.2annoy常见API整理

- AnnoyIndex(f, metric)返回可读写的新索引，用于存储f维度向量。metric 是"angular"，"euclidean"，"manhattan"，"hamming"，或"dot"。

- a.add_item(i, v)用于给索引添加向量v

- a.build(n_trees)用于构建 n_trees 的森林。查询时，树越多，精度越高。在调用build后，无法再添加任何向量。

- a.save(fn, prefault=False)将索引保存到磁盘。保存后，不能再添加任何向量。

- a.load(fn, prefault=False)从磁盘加载索引。如果prefault设置为True，它将把整个文件预读到内存中。默认值为False。

- a.unload() 释放索引。

- a.get_nns_by_item(i, n, search_k=-1, include_distances=False)返回第i 个item的n个最近邻的item。在查询期间，它将检索多达search_k（默认n_trees * n）个点。search_k为您提供了更好的准确性和速度之间权衡。

- a.get_nns_by_vector(v, n, search_k=-1, include_distances=False)与上面的相同，但按向量v查询。

- a.get_distance(i, j)返回向量i和向量j之间的距离。注意：此函数用于返回平方距离。

- a.get_n_items() 返回索引中的向量数。

- a.get_n_trees() 返回索引中的树的数量。

- a.on_disk_build(fn) 用以在指定文件而不是RAM中建立索引（在添加向量之前执行，在建立之后无需保存）。

### 权衡

调整Annoy仅需要两个主要参数：树的数量 n_trees 和搜索期间要检查的节点的数量search_k。

- n_trees在构建索引期间提供该值，并且会影响构建时间和索引大小。较大的值将给出更准确的结果，但索引较大。

- search_k是在运行时提供的，并且会影响搜索性能。较大的值将给出更准确的结果，但返回时间将更长

### 效果对比图

![img](https://netease-we.feishu.cn/space/api/box/stream/download/asynccode/?code=6ca697394511e2df138f6a7f1de7985d_8f118824ce50c961_boxcntbCxhiy2y8gEQN2C3pbCRc_p6qlYcxm71XbPf1yV7SzA3PHbacjASLd)



























https://yongyuan.name/blog/ann-search.html

 

 