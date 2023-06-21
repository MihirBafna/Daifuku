'''
--------------------------------------------------------------------------------
    Adapted from Ruochi Zhang's FastProcess.py from the Higashi repository:
    https://github.com/ma-compbio/Higashi/blob/main/higashi/Fast_Process.py
--------------------------------------------------------------------------------
'''

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Pad
from tqdm import trange, tqdm
from scipy.sparse import csr_matrix
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

try:
	get_ipython()
	from tqdm.notebook import tqdm, trange
except:
	pass

import os 
import numpy as np
import pandas as pd
import math
import multiprocessing
import pickle

cpu_num = multiprocessing.cpu_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1]


class HiCDataset(Dataset):
    def __init__(self, config):
        self.contact_map_path = config["temp_dir"]
        self.dataset_path = config["data_dir"]
        self.file_list_path = config["file_list_path"]
        self.dataset_info = pd.DataFrame(pickle.load(open(self.dataset_path+"/label_info.pickle","rb")))
        self.chrom_list = config['chrom_list']
        self.num_cells =  len(self.dataset_info)
        self.num_chromosomes =  len(self.chrom_list)
        self.is_sparse = True if config["is_sparse"] == "True" else False

        self.pseudobulk_maps = self.gather_pseudobulk_maps()

    def __len__(self):
        return self.num_cells*self.num_chromosomes

    def __getitem__(self, idx):
        ''' 
            chrom_idx \in [0,num_chromosomes - 1]
            cell_idx  \in [0, num_cells - 1]
        '''
        chrom_idx = idx // self.num_cells
        cell_idx = idx - chrom_idx * self.num_cells
        if self.is_sparse:
            contact_path = f"{self.contact_map_path}/sparse/chr{chrom_idx+1}_cell{str(cell_idx+1)}.npy"
        else:
            contact_path = f"{self.contact_map_path}/dense/chr{chrom_idx+1}_cell{str(cell_idx+1)}.npy"

        contact_map = np.load(contact_path, allow_pickle=True) # load contact maps for chromosome at index chrom_idx

        if self.is_sparse:
            # return contact_map_sparse
            contact_map_sparse = spy_sparse2torch_sparse(contact_map)
            return
        else:
            return torch.from_numpy(contact_map), self.pseudobulk_maps[chrom_idx]
                    
    def gather_pseudobulk_maps(self):
        pseudobulk_maps = []
        for chrom in self.chrom_list:
            map = np.load(f"{self.contact_map_path}/dense/{chrom}_pseudobulk.npy")
            map = torch.from_numpy(map).squeeze()
            pseudobulk_maps.append(map)
        return pseudobulk_maps


def get_config(config_path = "./config.json"):
	c = open(config_path,"r")
	return json.load(c)


def get_free_gpu():
	os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > ./tmp')
	memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
	if len(memory_available) > 0:
		max_mem = np.max(memory_available)
		ids = np.where(memory_available == max_mem)[0]
		chosen_id = int(np.random.choice(ids, 1)[0])
		print("setting to gpu:%d" % chosen_id)
		torch.cuda.set_device(chosen_id)
	else:
		return
	

def create_dir(config):
	temp_dir = config['temp_dir']
	if not os.path.exists(temp_dir):
		os.mkdir(temp_dir)
	
	raw_dir = os.path.join(temp_dir, "raw")
	if not os.path.exists(raw_dir):
		os.mkdir(raw_dir)
  
	bulk_dir = os.path.join(temp_dir, "bulk")
	if not os.path.exists(bulk_dir):
		os.mkdir(bulk_dir)
	
	
	# rw_dir = os.path.join(temp_dir, "rw")
	# if not os.path.exists(rw_dir):
	# 	os.mkdir(rw_dir)

	# embed_dir = os.path.join(temp_dir, "embed")
	# if not os.path.exists(embed_dir):
	# 	os.mkdir(embed_dir)
	
 
def spy_sparse2torch_sparse(data):
	"""
	:param data: a scipy sparse csr matrix
	:return: a sparse torch tensor
	"""
	samples=data.shape[0]
	features=data.shape[1]
	values=data.data
	coo_data=data.tocoo()
	indices=torch.LongTensor(np.array([coo_data.row,coo_data.col]))
	t=torch.sparse.FloatTensor(indices,torch.from_numpy(values).float(),[samples,features])
	return t


# Generate a indexing table of start and end id of each chromosome
def generate_chrom_start_end(config):
	# fetch info from config
	genome_reference_path = config['genome_reference_path']
	chrom_list = config['chrom_list']
	res = config['resolution']
	temp_dir = config['temp_dir']
	
	print ("generating start/end dict for chromosome")
	chrom_size = pd.read_table(genome_reference_path, sep="\t", header=None)
	chrom_size.columns = ['chrom', 'size']
	# build a list that stores the start and end of each chromosome (unit of the number of bins)
	chrom_start_end = np.zeros((len(chrom_list), 2), dtype='int')
	for i, chrom in enumerate(chrom_list):
		size = chrom_size[chrom_size['chrom'] == chrom]
		size = size['size'][size.index[0]]
		n_bin = int(math.ceil(size / res))
		chrom_start_end[i, 1] = chrom_start_end[i, 0] + n_bin
		if i + 1 < len(chrom_list):
			chrom_start_end[i + 1, 0] = chrom_start_end[i, 1]
	
	# print("chrom_start_end", chrom_start_end)
	np.save(os.path.join(temp_dir, "chrom_start_end.npy"), chrom_start_end)
	
	
def data2mtx(config, file, chrom_start_end, verbose, cell_id):
	if "header_included" in config:
		if config['header_included']:
			tab = pd.read_table(file, sep="\t")
		else:
			tab = pd.read_table(file, sep="\t", header=None)
			tab.columns = config['contact_header']
	else:
		tab = pd.read_table(file, sep="\t", header=None)
		tab.columns = config['contact_header']
	if 'count' not in tab.columns:
		tab['count'] = 1
	
	if 'downsample' in config:
		downsample = config['downsample']
	else:
		downsample = 1.0
		
	data = tab
	# fetch info from config
	res = config['resolution']
	chrom_list = config['chrom_list']
	
	data = data[(data['chrom1'] == data['chrom2']) & (np.abs(data['pos2'] - data['pos1']) >= 2500)]
	
	pos1 = np.array(data['pos1'])
	pos2 = np.array(data['pos2'])
	bin1 = np.floor(pos1 / res).astype('int')
	bin2 = np.floor(pos2 / res).astype('int')
	
	chrom1, chrom2 = np.array(data['chrom1'].values), np.array(data['chrom2'].values)
	count = np.array(data['count'].values)
	
	if downsample < 1:
		# print ("downsample at", downsample)
		index = np.random.permutation(len(data))[:int(downsample * len(data))]
		count = count[index]
		chrom1 = chrom1[index]
		bin1 = bin1[index]
		bin2 = bin2[index]
		
	del data
	
	m1_list = []
	for i, chrom in enumerate(chrom_list):
		mask = (chrom1 == chrom)
		size = chrom_start_end[i, 1] - chrom_start_end[i, 0]
		temp_weight2 = count[mask]
		m1 = csr_matrix((temp_weight2, (bin1[mask], bin2[mask])), shape=(size, size), dtype='float32')
		m1 = m1 + m1.T
		m1_list.append(m1)
		count = count[~mask]
		bin1 = bin1[~mask]
		bin2 = bin2[~mask]
		chrom1 = chrom1[~mask]
	
	return m1_list, cell_id





# Extra the data.txt table
# Memory consumption re-optimize
def extract_table(config):
	# fetch info from config
	data_dir = config['data_dir']
	temp_dir = config['temp_dir']
	chrom_list = config['chrom_list']
	file_list_path = config['file_list_path']
	if 'input_format' in config:
		input_format = config['input_format']
	else:
		input_format = 'higashi_v2'
	
	chrom_start_end = np.load(os.path.join(temp_dir, "chrom_start_end.npy"))
	if input_format == 'higashi_v1':
		print ("Sorry no higashi_v1")
		raise EOFError
			
	elif input_format == 'higashi_v2':
		print ("extracting from filelist.txt")
		with open(os.path.join(file_list_path), "r") as f:
			lines = f.readlines()
			filelist = [line.strip() for line in lines]
		bar = trange(len(filelist))
		mtx_all_list = [[0]*len(filelist) for i in range(len(chrom_list))]
		p_list = []
		pool = ProcessPoolExecutor(max_workers=cpu_num)
		for cell_id, file in enumerate(filelist):
			p_list.append(pool.submit(data2mtx, config, file, chrom_start_end, False, cell_id))
		
		
		for p in as_completed(p_list):
			mtx_list, cell_id = p.result()
			for i in range(len(chrom_list)):
				mtx_all_list[i][cell_id] = mtx_list[i]
			bar.update(1)
		bar.close()
		pool.shutdown(wait=True)
		for i in range(len(chrom_list)):
			np.save(os.path.join(temp_dir, "raw", "%s_sparse_adj.npy" % chrom_list[i]), mtx_all_list[i], allow_pickle=True)
		
	else:
		print ("invalid input format")
		raise EOFError
	
 
def unpack_contactmaps(config):
    '''
    Unpacking the higashi preprocessed contact maps into individual 2D npy arrays (for ease of torch dataset creation)
    '''
    contact_map_path = config["temp_dir"]
    chrom_list = config['chrom_list']
    is_sparse = True if config["is_sparse"] == "True" else False
    
    if is_sparse:
        unpack_path = os.path.join(contact_map_path,"sparse")
    else:
        unpack_path = os.path.join(contact_map_path,"dense")
        
    print(unpack_path)
    if not os.path.exists(unpack_path):
        os.mkdir(unpack_path)
    
    for chrom in tqdm(chrom_list, desc="unpacking contact maps"):
        chrom_contact_maps = np.load(f"{contact_map_path}/raw/{chrom}_sparse_adj.npy", allow_pickle=True) # load contact maps for chromosome at index chrom_idx
        for i, scipy_contact_map in enumerate(chrom_contact_maps):
            contact_map_dense = scipy_contact_map.todense()
            map_path = os.path.join(unpack_path, chrom+"_cell"+str(i+1)+".npy")
            if is_sparse:
                np.save(map_path,scipy_contact_map)
            else:
                np.save(map_path,contact_map_dense)
			
   
def get_knn(x, k):
    from sklearn.neighbors import KDTree
    kdt = KDTree(x, leaf_size=30)
    _ , indices = kdt.query(x, k=k)
    return indices


def generate_pseudobulk(config):
    higashi_embed_dir = config["higashi_embed_dir"]
    num_cells_pseudobulk = config["num_cells_pseudobulk"]
    contact_map_path = config["temp_dir"]
    chrom_list = config['chrom_list']
    higashi_embeddings = np.load(os.path.join(higashi_embed_dir,"embedding.npy"), allow_pickle=True)
    indices = get_knn(higashi_embeddings, k=num_cells_pseudobulk)
    
    unpack_path = os.path.join(contact_map_path,"bulk")
    print("Destination: "+unpack_path)
    if not os.path.exists(unpack_path):
        os.mkdir(unpack_path)
        
    for chrom in tqdm(chrom_list, desc="Generating pseudobulk contact maps"):
        chrom_contact_maps = np.load(f"{contact_map_path}/raw/{chrom}_sparse_adj.npy", allow_pickle=True) # load contact maps for chromosome at index chrom_idx
        chrom_contact_maps = chrom_contact_maps[indices]
        for i, neighbors in enumerate(chrom_contact_maps):
            pseudobulk_map = np.mean(np.array([sparse_map.todense() for sparse_map in neighbors]), axis=0)
            map_path = os.path.join(unpack_path, chrom+"_cell"+str(i+1)+"_pseudobulk.npy")
            np.save(map_path,pseudobulk_map)



def higashi_preprocess(config):
	create_dir(config)
	generate_chrom_start_end(config)
	extract_table(config)
	unpack_contactmaps(config)
	generate_pseudobulk(config)