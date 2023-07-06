# python endurance_task_analysis.py --file_path ../data/Annotations/SILVER_w:_r4_results.csv
import numpy as np
import pandas as pd
import krippendorff
from sklearn.metrics import cohen_kappa_score

import argparse
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("ggplot")


parser = argparse.ArgumentParser(description="endurance task analysis")
parser.add_argument("--file_path", type=str, default="./data/Annotations/SILVER_w:_r4_results.csv")
args = parser.parse_args()


def extract_endu_result(file_path):
	df = pd.read_csv(file_path)
	df = df.loc[:, ["WorkerId", "Input.instancejson", "Answer.score_0", "Answer.score_1", "Answer.score_2", "Answer.score_3"]]
	df = df.reset_index(drop=True)
	return df


def filter_workers(df):
	ls = [list(df.groupby("WorkerId").groups[i]) for i in df.groupby("WorkerId").groups if len(list(df.groupby("WorkerId").groups[i])) == 10]
	ls.sort()
	return ls


def CK_score_collector(answer_idx, df, worker_list):
	ck_scores = []

	for i in range(len(worker_list)):
		for j in range(len(worker_list)):
			ck = cohen_kappa_score(df.loc[worker_list[i], answer_idx], 
								   df.loc[worker_list[j], answer_idx], 
								   weights='linear')
			ck_scores.append(ck)

	ck_scores = np.array(ck_scores).reshape(len(worker_list), len(worker_list))
	return ck_scores


def single_CK_viz(df, worker_list):
	num_worker = len(worker_list)
	fig = plt.figure(figsize=(22, 5))
	fontsize=14

	for i, ans in enumerate(["Answer.score_0", "Answer.score_1", "Answer.score_2", "Answer.score_3"]):
		data = CK_score_collector(ans, df, worker_list)
		ax = fig.add_subplot(1, 4, i+1)
		sns.heatmap(data, annot=True, fmt=".3f", cmap="YlGnBu", mask=np.tril(np.ones((num_worker, num_worker)), k=0), ax=ax, linewidths=.5)
		ax.set_title(ans, fontsize=fontsize)
		# plt.xticks(np.arange(num_worker)+0.5, ["S21", "S22", "S23", "S31", "S32", "S41", "S42", "S43"], fontsize=fontsize-4)
		# plt.yticks(np.arange(num_worker)+0.5, ["S21", "S22", "S23", "S31", "S32", "S41", "S42", "S43"], fontsize=fontsize-4)
		plt.xticks(np.arange(num_worker)+0.5, [f"Worker_{i+1}" for i in range(num_worker)], fontsize=fontsize-4)
		plt.yticks(np.arange(num_worker)+0.5, [f"Worker_{i+1}" for i in range(num_worker)], fontsize=fontsize-4)
		ax.set_xlabel("Worker ID", fontsize=fontsize)
		ax.set_ylabel("Worker ID", fontsize=fontsize)
		
	plt.tight_layout()
	plt.show()
	# plt.savefig('/content/drive/MyDrive/GEM_Lining/heat_S_r4.pdf')


# Concate 4 scores
def all_scores(df, idx_list):
	all_scores = []
	for i in range(4):
		all_scores.extend(df.loc[idx_list, "Answer.score_{}".format(i)])
	return all_scores


def CK_score_collector_concate_omit(df, worker_list):
	ck_scores = []

	for i in range(len(worker_list)):
		for j in range(len(worker_list)):
			A_scores = all_scores(df, worker_list[i])
			B_scores = all_scores(df, worker_list[j])
			A_scores_omit = all_scores(df, worker_list[i][2:])
			B_scores_omit = all_scores(df, worker_list[j][2:])
			
			ck_concate = cohen_kappa_score(A_scores,
										   B_scores, 
										   weights='linear')
			ck_omit = cohen_kappa_score(A_scores_omit,
										B_scores_omit, 
										weights='linear')
			ck_scores.append(ck_concate)
			ck_scores.append(ck_omit)

	return (np.array(ck_scores[::2]).reshape(len(worker_list), len(worker_list)), np.array(ck_scores[1::2]).reshape(len(worker_list), len(worker_list)))


def concate_CK_viz(df, worker_list):
	num_worker = len(worker_list)
	fig = plt.figure(figsize=(5, 8))
	fontsize=14

	for i, ans in enumerate(["Concatenate", "Omit first 2 HITs"]):
		data = CK_score_collector_concate_omit(df, worker_list)
		ax = fig.add_subplot(2, 1, i+1)
		sns.heatmap(data[i], annot=True, fmt=".3f", cmap="YlGnBu", mask=np.tril(np.ones((num_worker, num_worker)), k=0), ax=ax, linewidths=.5)
		ax.set_title(ans, fontsize=fontsize)
		# plt.xticks(np.arange(num_worker)+0.5, ["S21", "S22", "S23", "S31", "S32", "S41", "S42", "S43"], fontsize=fontsize-4)
		# plt.yticks(np.arange(num_worker)+0.5, ["S21", "S22", "S23", "S31", "S32", "S41", "S42", "S43"], fontsize=fontsize-4)
		plt.xticks(np.arange(num_worker)+0.5, [f"Worker_{i+1}" for i in range(num_worker)], fontsize=fontsize-4)
		plt.yticks(np.arange(num_worker)+0.5, [f"Worker_{i+1}" for i in range(num_worker)], fontsize=fontsize-4)
		ax.set_xlabel("Worker ID", fontsize=fontsize)
		ax.set_ylabel("Worker ID", fontsize=fontsize)
		
	plt.tight_layout()
	plt.show()
	# plt.savefig('/content/drive/MyDrive/GEM_Lining/heat_concate_S_r4.pdf')


def krippendorff_alpha(df, worker_list):
	scores_list = [all_scores(df, i) for i in worker_list]
	all_data = pd.DataFrame(scores_list)

	print("Krippendorff's alpha for nominal metric: ", krippendorff.alpha(reliability_data=all_data, level_of_measurement='nominal'))
	print("Krippendorff's alpha for interval metric: ", krippendorff.alpha(reliability_data=all_data))
	print("Krippendorff's alpha for ordinal metric: ", krippendorff.alpha(reliability_data=all_data, level_of_measurement='ordinal'))


def main():
	df = extract_endu_result(args.file_path)
	worker_list = filter_workers(df)
	single_CK_viz(df, worker_list)
	concate_CK_viz(df, worker_list)
	krippendorff_alpha(df, worker_list)


if __name__ == '__main__':
	main()