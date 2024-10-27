# Pose and Joint-Aware Action Recognition

Code and Pre-processed data for the paper Pose and Joint-Aware Action Recognition accepted to WACV 2022

[[`Paper`](https://openaccess.thecvf.com/content/WACV2022/papers/Shah_Pose_and_Joint-Aware_Action_Recognition_WACV_2022_paper.pdf)] [[`Video`](https://youtu.be/BqaOlF_LOMA)]

## Set-up environment
- Tested with Python Version : 3.7.11
  

Follow one of the following to set up the environment:
- A) Install from conda environment : `conda env create -f environment.yml`
- B) The code mainly requires the following packages : torch, torchvision, puytorch 
  - Install one package at a time :
  - `conda create -n pose_action python=3.7`
  - `conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge`
  - `pip install opencv-python matplotlib wandb tqdm joblib scipy scikit-learn`
- C) Make an account on wandb and make required changes to `train.py` L36


## Prepare data
- `mkdir data`
- `mkdir metadata`

```xml
----data
	----JHMDB
		----openpose_COCO_3
			---....npy
			---....npy
			---....npy
	----HMDB51
		----openpose_COCO_3
			---....npy
			---....npy
			---....npy

----metadata
	----JHMDB
		----.pkl
		----.pkl
		----.pkl
	----HMDB51
		----.pkl
		----.pkl
		----.pkl
```



- Download data from [here](http://www.cis.jhu.edu/~ashah/PoseAction/data/). Extract the tar files with folder structure `data/$dataset/openpose_COCO_3/`
- Download metadata from [here](http://www.cis.jhu.edu/~ashah/PoseAction/metadata.tar.gz). Extract the tar files to `data/metadata`

## Training scripts
- Example : `bash sample_scripts/hmdb.sh`
- Example : `bash sample_scripts/jhmdb.sh`
- Example : `bash sample_scripts/le2i.sh`
- Raw heatmaps

We also provide raw heatmaps [here](https://1drv.ms/u/s!AlAjgCeVY_IrgY40FMWKAsiO5-Opmw?e=N8e4A6). OpenPose was used to extract these. Please take a look at function `final_extract_hmdb` in `utils.py` for an example function to extract pose data. 

## Citation
If you find this repository useful in your work, please cite us! 
```
@InProceedings{Shah_2022_WACV,
    author    = {Shah, Anshul and Mishra, Shlok and Bansal, Ankan and Chen, Jun-Cheng and Chellappa, Rama and Shrivastava, Abhinav},
    title     = {Pose and Joint-Aware Action Recognition},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2022},
    pages     = {3850-3860}
}
```





#### 本地到OSS [恒源云](https://www.gpushare.com/docs/data/upload/)

进入链接，里面有OSS命令上传数据，点击OSS命令安装然后下载exe文件改名为oss.exe

本地电脑Windows PowerShell，进入存放下载的oss.exe的目录

```sh
cd H:\
.\oss.exe login
Username:15072376330
Password:lzy24324615695
成功登录
.\oss.exe cp dataset.zip oss://
```

#### OSS到实例

启动实例`JupyterLab`，将OSS的数据传到服务器的 /hy-tmp中

```sh
oss login
Username:150.....
Password:lzy.....
成功登录
oss cp oss://dataset.zip /hy-tmp/
cd /hy-tmp/
```

#### 解压

```sh
unzip dataset.zip
```

#### 原始PoseAction改动

```java
git clone https://github.com/anshulbshah/PoseAction.git
```

修改opt中的--name

注释wandb，在train和trains中

HMDB51的文件名没有-

HMDB51训练脚本batchsize128太大，改为32

models.py中打印参数修改 print('Number of parameters requiring grad : {} '.format(count_parameters(enc)))



#### 问题

libgthread-2.0.so.0: cannot open shared object file: No such file or directory

```bash
sudo apt-get install libglib2.0-0
```



#### 本地连接服务器终端

在Pycharm的Tools中选择Start SSH session [参考](https://blog.csdn.net/qq_45100200/article/details/130355935?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171983704416800188557392%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=171983704416800188557392&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-5-130355935-null-null.142^v100^pc_search_result_base6&utm_term=pycharm%E8%BF%9C%E7%A8%8B%E6%9C%8D%E5%8A%A1%E5%99%A8%E7%BB%88%E7%AB%AF&spm=1018.2226.3001.4187)

#### 
