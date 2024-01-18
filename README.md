##这是异构图神经网络用于解决语句正误预测问题的代码包，其结构分为：  
1. read_data.py  
这个文件是用于读取pkl数据，并转换成可以训练的异构图列表的。  
2. model.py  
这个文件是定义了GNN-FiLM模型，能够直接调用到主函数中的。  
3. main.py  
主要用于训练，以及预测的模型，具体流程是先调用读取数据和转换数据的函数，再调用模型构造函数，最后进行训练，定义了每个图的训练函数和整体的训练过程。  

使用方法：  
在本项目根目录下创建并激活新的虚拟环境  
`conda create -n torch_geometric python=3.9`  
`conda activate torch_geometric`  
`pip install -r requiements.txt`  
`python main.py`  