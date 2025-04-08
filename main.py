import argparse
from models import MODEL_REGISTRY, MLModel, PLBaseModel, DLModel
from omegaconf import OmegaConf
from utils import read_data,distinguish,save_data
import numpy as np
import os
import pandas as pd
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GBDT', help='选择模型')
    parser.add_argument('--data', type=str, default='data1.xlsx', help='数据文件路径')
    parser.add_argument('--xyz', type=str, default='xyz', help='是否包含坐标')
    parser.add_argument('--use_loc', type=bool, default=False, help='是否使用坐标')
    parser.add_argument('--gnn', type=int, default=0,choices=[0,1,2],help='none 0, gnn 1 and cnn folown by location 2')
    parser.add_argument('--config', type=str, default='./configs/cnn3d.yaml', help='模型配置')
    parser.add_argument('--save', type=bool, default=True, help='保存结果的路径')
    return parser.parse_args()

def main():
    args = parse_args()
    if os.path.exists(args.config):
        config = OmegaConf.load(args.config)
        args.model = config.name
        options = config.params
    else:
        options = None

    model_type = distinguish(args.model)
    if model_type == "machine":
        if args.model not in MODEL_REGISTRY:
            raise ValueError(f"模型 {args.model} 不在支持列表中: {list(MODEL_REGISTRY.keys())}")
        ModelClass = MODEL_REGISTRY[args.model]
        if options is not None:
            print(f"使用配置：{args.config}")
            model = ModelClass(**options)
        else:
            model = ModelClass()
        print(f"使用模型：{model.__class__.__name__}")
    else:
        model = PLBaseModel(config)
        print(f"使用模型：{model.__class__.__name__}")

    # 读取数据
    x_train,y_train,x_val,y_val,x_test,scaler,loc_data,coloms = read_data(args.data,args.xyz,args.use_loc,args.model)
    # 训练模型


    # model_type = distinguish(args.model)
    if model_type == "machine":
        model = MLModel(model,x_train,y_train,x_val,y_val,x_test)
    else:
        model = DLModel(model,x_train,y_train,x_val,y_val,x_test)

    model.train()
    model.evaluate()
    print("训练完成")

    if args.save:
        save_data(model,scaler,loc_data,coloms,f'{args.model}_result.xlsx')

if __name__ == '__main__':
    main()
