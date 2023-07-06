# python stat_test.py --round_list ../data/Worker_status/qual_task_v2_r1.txt --gold_list ../data/Worker_status/qual_task_v2_r1_GOLD.txt --silver_list ../data/Worker_status/qual_task_v2_r1_SILVER.txt
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
np.random.seed(1234)


parser = argparse.ArgumentParser(description="statistical test using bootstrap")
parser.add_argument("--round_list", type=str, default="../data/Worker_status/qual_task_v2_r1.txt")
parser.add_argument("--gold_list", type=str, default="../data/Worker_status/qual_task_v2_r1_GOLD.txt")
parser.add_argument("--silver_list", type=str, default="../data/Worker_status/qual_task_v2_r1_SILVER.txt")
parser.add_argument("--score4_list", type=str, default="../data/Worker_status/endu_task_score_4.txt")
parser.add_argument("--n", type=int, default=50)
parser.add_argument("--num_iter", type=int, default=2000)
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

def main():
	worker_Ids = read_txt(args.round_list)
	num_worker = len(worker_Ids)
	gold_list, silver_list = read_txt(args.gold_list), read_txt(args.silver_list)
	q_pass_list = gold_list + silver_list
	e_pass_list = read_txt(args.score4_list)

	df = pd.DataFrame()
	df["Workers"] = worker_Ids
	df["q_pass"] = df["Workers"].map(lambda x: element_mapper(x, q_pass_list))
	df["e_pass"] = df["Workers"].map(lambda x: element_mapper(x, e_pass_list))

	prob_q_pass = []
	prob_e_pass = []
	print("Bootstrap begins ...")
	for i in tqdm(range(args.num_iter)):
		idxs = np.random.choice(num_worker, args.n, replace=True)
		new_df = df.iloc[idxs]
		num_q_pass = new_df["q_pass"].sum()
		num_e_pass = new_df["e_pass"].sum()

		prob_q_pass.append(num_q_pass / args.n)
		prob_e_pass.append(num_e_pass / args.n)

	print("For Round {}:".format(args.round_list.split("/")[-1][-5]))
	print("The observed probability of passing Qualification Task: {}".format(np.sum(df["q_pass"])/num_worker))
	print("The observed probability of passing All Tasks: {}\n".format(np.sum(df["e_pass"])/num_worker))

	print("After bootstrap of {} times:".format(args.num_iter))
	print("The mean of the probability of passing Qualification Task: %.4f" % np.mean(prob_q_pass))
	print("The std of the probability of passing Qualification Task: %.4f\n" % np.std(prob_q_pass))

	print("The mean of the probability of passing All Tasks: %.4f" % np.mean(prob_e_pass))
	print("The std of the probability of passing All Tasks: %.4f\n" % np.std(prob_e_pass))

if __name__ == '__main__':
	main()