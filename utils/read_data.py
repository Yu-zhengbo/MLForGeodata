import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import copy
from einops import rearrange
def read_data(file_path,xyz='xyz',use_loc=False,model='cnn1d',fixed_split=False,test_size=0.2,random_state=42, scaler='standard'):
    if model in ['cnn2d','cnn3d']:
        return read_data_2d(file_path,xyz=xyz,test_size=test_size,random_state=random_state, scaler=scaler, neighbour_size=5 if model == 'cnn2d' else 3, mode=model)
    else:
        return read_data_v1(file_path,xyz=xyz,use_loc=use_loc,fixed_split=fixed_split,test_size=test_size,random_state=random_state, scaler=scaler)

def read_data_v1(file_path,xyz='xyz',use_loc=False,fixed_split=False,test_size=0.2,random_state=42, scaler='standard'):
    # input : file_path : str, the path of the excel file.  label = -1 means unlabeled data. fixed_split = True
    #         means the labeled data is split into train and val.
    """
        X	             Y	   Z          AU	 F4(HG-SB)	     QW	      ILR近矿晕	     label    fixed_split
        34573210	3870640	  2790	87.72789122	-0.36697097	302.4815455	     0.92692401	  0           1
        34573210	3871070	  3190	13.38004617	-0.40075837	17.24309652 	-1.11434574	  1           1
        34573210	3871070	  3200	13.37965787	-0.40087193	17.244353	    -1.11427288	  1           1
        34573210	3871070	  3210	13.89582037	-0.69381409	16.46651362 	-1.25393306	  1           0
        34573210	3871080	  3220	19.37533062	0.03639883	22.1374341  	-1.23515603	  1           1
        34573210	3871080	  3230	17.50300651	-0.01603275	19.13785262	    -1.39808942	  1           0
        34573210	3871080   3240	17.33008903	-0.14142068	21.16105771     -1.40422619	 -1           1
        34573210	3871080	  3250	17.50050271	-0.01591997	19.1317323	    -1.39821487	 -1           1
    """
    # output : x_train,y_train,x_val,y_val,x_test,scaler 
    #          x_train: N,C;
    #          y_train: N,1;
    #          x_val: N,C;
    #          y_val: N,1;
    #          x_test: N,C;
    #          scaler: StandardScaler or MinMaxScaler or None. scaler is used to rescale the data and generate the final result.
    data = pd.read_excel(file_path)
    colomns = data.columns.tolist()[:-2 if fixed_split else -1]
    data = np.array(data)
    if not use_loc:
        loc_data = data[:,:len(xyz)]
        data = data[:,len(xyz):]


    if scaler =='standard':
        scaler = StandardScaler()
    elif scaler =='minmax':
        scaler = MinMaxScaler()
    else:
        scaler = None

    if scaler is not None:
        data[:,:-2 if fixed_split else -1] = scaler.fit_transform(data[:,:-2 if fixed_split else -1])

    data_unlabeled = data[data[:,-2 if fixed_split else -1] == -1]
    loc_data_unlabeled = loc_data[data[:,-2 if fixed_split else -1] == -1]
    data_labeled = data[data[:,-2 if fixed_split else -1] != -1]


    if fixed_split:
        split = data_labeled[:,-1]
        data_labeled = data_labeled[:,:-1]
        train_mask = split == 1
        val_mask = split == 0
        x_train,y_train = data_labeled[train_mask][:,:-1],data_labeled[train_mask][:,-1]
        x_val,y_val = data_labeled[val_mask][:,:-1],data_labeled[val_mask][:,-1]
        x_test = data_unlabeled[:,:-2]
    else:
        X,Y = data_labeled[:,:-1],data_labeled[:,-1]
        x_train,x_val,y_train,y_val = train_test_split(X,Y,test_size=test_size, random_state=random_state)
        x_test = data_unlabeled[:,:-1]


    return x_train,y_train,x_val,y_val,x_test,scaler,loc_data_unlabeled,colomns+['pred','score']


def save_data(model,scaler,loc_data,coloms,save_path):
    x_test,y_pred = model.predict()
    _,y_score = model.predict_proba()
    if len(x_test.shape) > 2:
        x_test = x_test.squeeze(dim=1).numpy()
    if model.model.name == 'cnn3d':
        k_size = x_test.shape[-1]
        x_test = x_test[:,:,k_size//2,k_size//2,k_size//2]
        x_test = scaler.inverse_transform(x_test)[:,3:]
        y_pred = y_pred[:,k_size//2,k_size//2,k_size//2]
        y_score = y_score[:,:,k_size//2,k_size//2,k_size//2]
    elif model.model.name == 'cnn2d':
        k_size = x_test.shape[-1]
        x_test = x_test[:,:,k_size//2,k_size//2]
        x_test = scaler.inverse_transform(x_test)[:,2:]
        y_pred = y_pred[:,k_size//2,k_size//2]
        y_score = y_score[:,:,k_size//2,k_size//2]
    else:
        x_test = scaler.inverse_transform(x_test)
    saved_data = np.concatenate((loc_data,x_test,y_pred.reshape(-1,1),y_score[:,1].reshape(-1,1)),axis=1)
    saved_data = pd.DataFrame(saved_data,columns=coloms)
    saved_data.to_excel(save_path,index=False)
    print("结果已保存至%s"%save_path)


def read_data_2d(file_path,xyz='xy',test_size=0.2,random_state=42, scaler='standard',neighbour_size=5,mode='cnn2d'):

    data = pd.read_excel(file_path)
    colomns = data.columns.tolist()[:-1]
    data = np.array(data)
    
    loc_data = data[:,:len(xyz)]
    loc_data_unlabeled = loc_data[data[:,-1] == -1]
    
    if scaler =='standard':
        scaler = StandardScaler()
    elif scaler =='minmax':
        scaler = MinMaxScaler()
    else:
        scaler = None


    data = get_neighbour_data(data, neighbour_size=neighbour_size, mode=mode)

    if scaler is not None:
        N = data.shape[0]
        # data[:,:-1] = scaler.fit_transform(data[:,:-1])
        data = data.reshape(N,-1,neighbour_size**2 if mode == 'cnn2d' else neighbour_size**3)
        data = rearrange(data,'N C K -> (N K) C')
        data[:,:-1] = scaler.fit_transform(data[:,:-1])
        data = rearrange(data,'(N K) C -> N C K',N=N,K=neighbour_size**2 if mode == 'cnn2d' else neighbour_size**3)
        if mode == 'cnn2d':
            data = data.reshape(N,-1,neighbour_size,neighbour_size)
        else:
            data = data.reshape(N,-1,neighbour_size,neighbour_size,neighbour_size)

    if mode == 'cnn2d':
        data_unlabeled = data[data[:,-1,neighbour_size//2,neighbour_size//2] == -1]
        data_labeled = data[data[:,-1,neighbour_size//2,neighbour_size//2] != -1]
    else:
        data_unlabeled = data[data[:,-1,neighbour_size//2,neighbour_size//2,neighbour_size//2] == -1]
        data_labeled = data[data[:,-1,neighbour_size//2,neighbour_size//2,neighbour_size//2] != -1]

    X,Y = data_labeled[:,:-1],data_labeled[:,-1]
    x_train,x_val,y_train,y_val = train_test_split(X,Y,test_size=test_size, random_state=random_state)
    x_test = data_unlabeled[:,:-1]

    return x_train,y_train,x_val,y_val,x_test,scaler,loc_data_unlabeled,colomns+['pred','score']

def get_nearst_position_k_index(pos, positions,k=1):
    distances = np.sum((positions - pos)**2, axis=1)
    return np.argsort(distances)[:k]
    

def get_neighbour_data(data, neighbour_size=5, mode='cnn2d'):
    """
    :param data: N,C
    :param neighbour_size: int
    :return: N,C,neighbour_size,neighbour_size or N,C,neighbour_size,neighbour_size,neighbour_size
    """
    if mode == 'cnn2d':
        neighbour_data = np.zeros((data.shape[0], data.shape[1], neighbour_size, neighbour_size))
        for i in range(data.shape[0]):
            pos_temp = data[i, :2]
            neighbour_index = get_nearst_position_k_index(pos_temp, data[:, :2], k=neighbour_size**2)
            data_voxel = data[neighbour_index, :].transpose(1, 0).reshape(-1,neighbour_size, neighbour_size)
            data_voxel[:2] -= pos_temp[:,None,None]
            data_voxel[:,0,0],data_voxel[:,neighbour_size//2,neighbour_size//2] = data_voxel[:,neighbour_size//2,neighbour_size//2],data_voxel[:,0,0]
            neighbour_data[i, :, :, :] = data_voxel
    elif mode == 'cnn3d':
        neighbour_data = np.zeros((data.shape[0], data.shape[1], neighbour_size, neighbour_size, neighbour_size))
        for i in range(data.shape[0]):
            pos_temp = data[i, :3]
            neighbour_index = get_nearst_position_k_index(pos_temp, data[:, :3], k=neighbour_size**3)
            data_voxel = data[neighbour_index, :].transpose(1, 0).reshape(-1,neighbour_size, neighbour_size, neighbour_size)
            data_voxel[:3] -= pos_temp[:,None,None,None]
            # data_voxel[:,0,0,0],data_voxel[:,neighbour_size//2,neighbour_size//2,neighbour_size//2] = data_voxel[:,neighbour_size//2,neighbour_size//2,neighbour_size//2],data_voxel[:,0,0,0]
            # 会发生浅拷贝
            center_point = copy.deepcopy(data_voxel[:,neighbour_size//2,neighbour_size//2,neighbour_size//2])
            data_voxel[:,neighbour_size//2,neighbour_size//2,neighbour_size//2] = data_voxel[:,0,0,0]
            data_voxel[:,0,0,0] = center_point
            neighbour_data[i, :, :, :, :] = data_voxel
    else:
        raise ValueError("mode should be '2d' or 'cnn3d'")
    
    return neighbour_data
        