# python perm_test.py --worker_list ../data/Worker_status/all_workers.txt --gold_list ../data/Worker_status/qual_task_v2_all_GOLD.txt --silver_list ../data/Worker_status/qual_task_v2_all_SILVER.txt
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm
np.random.seed(1234)


parser = argparse.ArgumentParser(description="permutation test")
parser.add_argument("--worker_list", type=str, default="../data/Worker_status/all_workers.txt")
parser.add_argument("--gold_list", type=str, default="../data/Worker_status/qual_task_v2_all_GOLD.txt")
parser.add_argument("--silver_list", type=str, default="../data/Worker_status/qual_task_v2_all_SILVER.txt")
parser.add_argument("--score4_list", type=str, default="../data/Worker_status/endu_task_score_4.txt")
parser.add_argument("--n_round", type=int, default=4)
args = parser.parse_args()

def read_txt(file_path):
	with open(file_path, 'r') as f:
		content = [i.strip() for i in f.readlines()]
	return content

def element_mapper(element, ls):
	if element in ls:
		return 1
	else:
		return 0

def statistic(x, y):
	return np.mean(x) - np.mean(y)

def permutation_test(a,b):

	sizea, sizeb = len(a), len(b)
	s = -np.abs(np.array(a).mean() - np.array(b).mean())
	data = np.concatenate((a,b))
	index = np.arange(len(data))

	all_s = []
	test_size = 100000
	for i in range(test_size):
		np.random.shuffle(index)
		aidx, bidx = index[:sizea], index[sizea:]
		all_s.append(data[aidx].mean() - data[bidx].mean())

	std = np.std(all_s)
	percentage = norm.cdf(s, loc=0, scale=std)*2

	return percentage

def main():
	worker_Ids = read_txt(args.worker_list)
	gold_list, silver_list = read_txt(args.gold_list), read_txt(args.silver_list)
	q_pass_list = gold_list + silver_list
	e_pass_list = read_txt(args.score4_list)

	df = pd.DataFrame()
	df["Workers"] = worker_Ids
	df["q_pass"] = df["Workers"].map(lambda x: element_mapper(x, q_pass_list))
	df["e_pass"] = df["Workers"].map(lambda x: element_mapper(x, e_pass_list))

	rounds_data_q = np.array(df["q_pass"]).reshape(args.n_round, int(df.shape[0]/args.n_round))
	rounds_data_e = np.array(df["e_pass"]).reshape(args.n_round, int(df.shape[0]/args.n_round))

	for data in [rounds_data_q, rounds_data_e]:
		for i in range(args.n_round-1):
			for j in range(i+1, args.n_round):
				x = data[i]
				y = data[j]
				pvalue = permutation_test(x,y)
				print("The percentage for Round {} and Round {}: {}".format(i+1, j+1, round(pvalue, 5)))
		print("Permutation test finished!")

if __name__ == '__main__':	
	main()