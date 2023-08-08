# python qualification_task_analysis.py --file_path ../data/Annotations/Batch_4764797_batch_results.csv
import os
import json
import jsonlines
import pandas as pd   
import numpy as np
import krippendorff
import time
import csv
import dateutil.parser
import dateutil.tz
import html
import sys
import argparse
import matplotlib.pyplot as plt
from typing import Dict, List, NamedTuple
from datetime import datetime

# To avoid csv file exceed the field limit
csv.field_size_limit(sys.maxsize)


parser = argparse.ArgumentParser(description="qualification task analysis")
parser.add_argument("--file_path", type=str, default="../data/Annotations/Batch_4764797_batch_results.csv")
args = parser.parse_args()


class Answers(NamedTuple):
	understandability: bool
	compactness: bool
	grammaticality: bool
	coherence: bool
	faithfulness: bool
	saliency: bool


class Annotation(NamedTuple):
	answers_list: List[Answers]
	instructions_summary: str
	feedback: str
	events: List[Dict]
	work_time: int
	hit_id: str
	worker_id: str
	accept_time: datetime
	submit_time: datetime
	metadata: Dict


_tzinfos = {
	"PDT": dateutil.tz.gettz("America/Los_Angeles"),
}


def _load_annotations(input_file: str) -> List[Annotation]:
	'''
	from Dan's post-processing: 
	https://github.com/GEM-benchmark/gem-v2-human-eval-analysis/blob/main/gem/qualification/pilot/postprocess.py
	'''
	annotations = []
	with open(input_file, "r") as f:
		reader = csv.reader(f)
		for i, row in enumerate(reader):
			if i == 0:
				indices = {column: j for j, column in enumerate(row)}
			else:
				# Load the results from each of the qualification questions.
				# There are 3 of them (0 indexed)
				answers_list = []
				for j in range(3):
					answers_list.append(
						Answers(
							row[indices[f"Answer.understandability_{j}"]].lower() == "yes",
							row[indices[f"Answer.compactness_{j}"]].lower() == "yes",
							row[indices[f"Answer.grammaticality_{j}"]].lower() == "yes",
							row[indices[f"Answer.coherence_{j}"]].lower() == "yes",
							row[indices[f"Answer.faithfulness_{j}"]].lower() == "yes",
							row[indices[f"Answer.saliency_{j}"]].lower() == "yes",
						)
					)
				
				# Remove instructions_summary & feedback that does not exist in columns
				instructions_summary = False
				feedback = False

				# Collect other information
				events = json.loads(html.unescape(row[indices["Answer.events"]]))
				work_time = int(row[indices["WorkTimeInSeconds"]])
				hit_id = row[indices["HITId"]]
				worker_id = row[indices["WorkerId"]]
				accept_time = dateutil.parser.parse(row[indices["AcceptTime"]], tzinfos=_tzinfos).astimezone(dateutil.tz.UTC)
				submit_time = dateutil.parser.parse(row[indices["SubmitTime"]], tzinfos=_tzinfos).astimezone(dateutil.tz.UTC)
				metadata = json.loads(row[indices["Answer.metadata"]])

				annotations.append(
					Annotation(
						answers_list,
						instructions_summary,
						feedback,
						events,
						work_time,
						hit_id,
						worker_id,
						accept_time,
						submit_time,
						metadata,
					)
				)
	return annotations


def extract_data(file_path):
	annotations = _load_annotations(file_path)
	questions = ["understandability", "compactness", "grammaticality", "coherence", "faithfulness", "saliency"]

	annotation_list = []
	index_ids = []
	for a in annotations:
	  # Use worker_id as index of dataframe
	  index_ids.append(a.worker_id)
	  annotation_list.append([a.answers_list[i][q] for i in range(3) for q in range(len(questions))])
	df = pd.DataFrame(annotation_list, columns = [qtext + str(i) for i in range(3) for qtext in questions], index = index_ids)
	return df


def attention_check(df):
	# For summarization, the last 3 questions should be 0. Otherwise block.
	print("%d total responses" % len(df))
	df['label'] = ((df['coherence2']==False)&(df['faithfulness2']==False)&(df['saliency2']==False)).map({True:'GOLD',False:'BLO'})
	print("%d passed the attention check" % len(df[(df['coherence2']==False)&(df['faithfulness2']==False)&(df['saliency2']==False)]))
	print("Attention check pass rate: " + str(len(df[df.label == "GOLD"])/len(df)))
	return df


def worker_categorization(df):
	# scores all questions
	# coherence2, faithfulness2, saliency2 are not included (since they are represented by "label" column as attention check)
	df['scores'] = (df['understandability0']).astype(int) + (df['compactness0']).astype(int) + (df['grammaticality0']).astype(int) + (df['coherence0']).astype(int) + (df['faithfulness0']).astype(int) + (df['saliency0']).astype(int) + (df['understandability1']).astype(int) + (-df['compactness1']).astype(int) + (df['grammaticality1']).astype(int) + (-df['coherence1']).astype(int) + (-df['faithfulness1']).astype(int) + (-df['saliency1']).astype(int) + (df['understandability2']).astype(int) + (df['compactness2']).astype(int) + (df['grammaticality2']).astype(int) + df['label'].map({'GOLD':0, 'BLO':-16})
	df.hist(column='scores', bins=34)
	plt.show()

	# maps all correct GOLD, all but 1 correct to SILVER, attention check passed to BRONZE, and attention check failed to BLOCK.
	# Note that GOLD, SILVER, and BRONZE also must have passed attention check or they would be negative
	df['label_all'] = ((df['scores']>14).astype(int) + (df['scores']>13).astype(int) + (df['scores']>0).astype(int)).map({3:'GOLD',2:'SILVER',1:'BRONZE',0:'BLOCK'})
	print(df['label_all'].value_counts())

	# df.to_csv("QUAL_RESULTS_SUMM_" + str(int(time.time())), index = False, header=True)
	return df


def krippendorff_alpha(df):
	# Calculate Krippendorff's alpha (before and after removing annotators who failed attention check)
	all_df = df.copy()
	pass_df = df[df['label_all'] != "BLOCK"]
	gold_df = df[df['label_all'] == "GOLD"]

	# delete columns of "label", "scores", "label_all"
	del all_df['label']
	del gold_df['label']
	del pass_df['label']
	del all_df['label_all']
	del gold_df['label_all']
	del pass_df['label_all']
	del all_df['scores']
	del gold_df['scores']
	del pass_df['scores']

	# calculate Krippendorff's alpha excluding the last 3 columns of attention check
	all_data = [[int(r) for r in ratings[:-3]] for ratings in all_df.values]
	print("Krippendorff's alpha with all evaluators (%d evaluators):" % df.shape[0])
	print("Krippendorff's alpha for nominal metric: ", krippendorff.alpha(reliability_data=all_data, level_of_measurement='nominal'))
	print("Krippendorff's alpha for interval metric: ", krippendorff.alpha(reliability_data=all_data))
	print("Krippendorff's alpha for ordinal metric: ", krippendorff.alpha(reliability_data=all_data, level_of_measurement='ordinal'))

	pass_data = [[int(r) for r in ratings[:-3]] for ratings in pass_df.values]
	print("="*30)
	print("Krippendorff's alpha with gold, silver, and bronze evaluators (%d evaluators):" % pass_df.shape[0])
	print("Krippendorff's alpha for nominal metric: ", krippendorff.alpha(reliability_data=pass_data, level_of_measurement='nominal'))
	print("Krippendorff's alpha for interval metric: ", krippendorff.alpha(reliability_data=pass_data))
	print("Krippendorff's alpha for ordinal metric: ", krippendorff.alpha(reliability_data=pass_data, level_of_measurement='ordinal'))

	gold_data = [[int(r) for r in ratings[:-3]] for ratings in gold_df.values]
	print("="*30)
	print("Krippendorff's alpha with gold evaluators (%d evaluators):" % gold_df.shape[0])
	if len(gold_data) > 1:
	  print("Krippendorff's alpha for nominal metric: ", krippendorff.alpha(reliability_data=gold_data, level_of_measurement='nominal'))
	  print("Krippendorff's alpha for interval metric: ", krippendorff.alpha(reliability_data=gold_data))
	  print("Krippendorff's alpha for ordinal metric: ", krippendorff.alpha(reliability_data=gold_data, level_of_measurement='ordinal'))
	else:
	  print("oops, not enough data!")


def main():
	df = extract_data(args.file_path)
	df = attention_check(df)
	df = worker_categorization(df)
	krippendorff_alpha(df)


if __name__ == '__main__':
	main()