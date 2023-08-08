# python reference_task_analysis.py --file_path ../data/Annotations/reference_based_results.csv
import numpy as np
import pandas as pd
import krippendorff
from sklearn.metrics import cohen_kappa_score

import argparse
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("ggplot")


parser = argparse.ArgumentParser(description="reference task analysis")
parser.add_argument("--file_path", type=str, default="../data/Annotations/reference_based_results.csv")
args = parser.parse_args()


def extract_ref_result(file_path):
	df = pd.read_csv(file_path)
	df = df.loc[:, ["WorkerId", "Input.instancejson", \
					"Answer.cand2ref0", "Answer.cand2ref1", "Answer.cand2ref2", "Answer.cand2ref3", \
					"Answer.ref2cand0", "Answer.ref2cand1", "Answer.ref2cand2", "Answer.ref2cand3"]]
	df = df.reset_index(drop=True)
	return df


def get_worker_info(df, num_worker, worker_reindex):
	all_workers = list(df.groupby("WorkerId").groups.keys())
	worker_ids = []
	for worker in all_workers:
		if len(df.groupby("WorkerId").groups[worker]) == num_worker:
			worker_ids.append(worker)

	worker_list = [df.groupby("WorkerId").groups[worker_ids[i]] for i in worker_reindex]
	return worker_ids, worker_list


def concate_scores(df, idx_list, keywords):
	all_scores = []
	for keyword in keywords:
		for i in range(4):
			all_scores.extend(df.loc[idx_list, "Answer."+keyword+"{}".format(i)])
	return all_scores


def CK_score_collector(df, worker_list):
	ck_holder = []

	for keyword in [["cand2ref"], ["ref2cand"], ["cand2ref", "ref2cand"]]:
		for i in range(len(worker_list)):
			for j in range(len(worker_list)):
				A_scores = concate_scores(df, worker_list[i], keyword)
				B_scores = concate_scores(df, worker_list[j], keyword)
				
				ck_score = cohen_kappa_score(A_scores,
											 B_scores, 
											 weights='linear')
				ck_holder.append(ck_score)

	return [np.array(ck_holder[:64]).reshape(len(worker_list), len(worker_list)), \
			np.array(ck_holder[64:128]).reshape(len(worker_list), len(worker_list)), \
			np.array(ck_holder[128:]).reshape(len(worker_list), len(worker_list))]


def pipeline_CK_viz(df, worker_list):
	num_worker = len(worker_list)
	fig = plt.figure(figsize=(16, 5))
	fontsize=14

	data_dict = {"cand2ref": CK_score_collector(df, worker_list)[0],
				 "ref2cand": CK_score_collector(df, worker_list)[1],
				 "all": CK_score_collector(df, worker_list)[2]}

	for i, type_ in enumerate(data_dict.keys()):
		data = data_dict[type_]
		ax = fig.add_subplot(1, 3, i+1)
		sns.heatmap(data, annot=True, fmt=".2f", cmap="YlGnBu", mask=np.tril(np.ones((num_worker, num_worker)), k=0), ax=ax, linewidths=.5)
		plt.xticks(np.arange(num_worker)+0.5, [f"W{i+1}" for i in range(num_worker)], fontsize=fontsize-4)
		plt.yticks(np.arange(num_worker)+0.5, [f"W{i+1}" for i in range(num_worker)], fontsize=fontsize-4)
		# plt.xticks(np.arange(num_worker)+0.5, ["S21", "S22", "S23", "S32", "S42", "S43", "G11", "G21"], fontsize=fontsize-4)
		# plt.yticks(np.arange(num_worker)+0.5, ["S21", "S22", "S23", "S32", "S42", "S43", "G11", "G21"], fontsize=fontsize-4)
		ax.set_title(type_, fontsize=fontsize)
		ax.set_xlabel("Worker ID", fontsize=fontsize)
			
	plt.tight_layout()
	plt.show()
	# plt.savefig('/content/drive/MyDrive/GEM_Lining/heat_ref.pdf')


def krippendorff_alpha(df, worker_list, worker_ids, worker_reindex, score_type_list):
	WorkerId_idx = [worker_ids[i] for i in worker_reindex]
	scores_list = [concate_scores(df, i, score_type_list) for i in worker_list]
	df_scores = pd.DataFrame(scores_list)
	df_scores.index = WorkerId_idx

	print("Krippendorff's alpha for nominal metric: ", krippendorff.alpha(reliability_data=df_scores, level_of_measurement='nominal'))
	print("Krippendorff's alpha for interval metric: ", krippendorff.alpha(reliability_data=df_scores))
	print("Krippendorff's alpha for ordinal metric: ", krippendorff.alpha(reliability_data=df_scores, level_of_measurement='ordinal'))


def main():
	worker_reindex = [0, 3, 6, 1, 5, 7, 2, 4]
	score_type_list = ["cand2ref", "ref2cand"]

	df = extract_ref_result(args.file_path)
	worker_ids, worker_list = get_worker_info(df, 30, worker_reindex)
	pipeline_CK_viz(df, worker_list)
	krippendorff_alpha(df, worker_list, worker_ids, worker_reindex, score_type_list)


if __name__ == '__main__':
	main()