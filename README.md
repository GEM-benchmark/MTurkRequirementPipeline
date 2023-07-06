# A Needle in a Haystack: An Analysis of High-Agreement Workers on MTurk for Summarization

## Abstract
> To prevent the costly and inefficient use of resources on low-quality annotations, we want a method for creating a pool of dependable annotators who can effectively complete difficult tasks, such as evaluating automatic summarization. Thus, we investigate the recruitment of high-quality Amazon Mechanical Turk workers via a two-step pipeline. We show that we can successfully filter out subpar workers before they carry out the evaluations and obtain high-agreement annotations with similar constraints on resources. Although our workers demonstrate a strong consensus among themselves and CloudResearch workers, their alignment with expert judgments on a subset of the data is not as expected and needs further training in correctness. This paper still serves as a best practice for the recruitment of qualified annotators in other challenging annotation tasks.

## Getting the code
You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/GEM-benchmark/MTurkRequirementPipeline.git
    
## Citation

```BibTeX
@article{Zhang2022NeedleIA,
      title={A Needle in a Haystack: An Analysis of High-Agreement Workers on MTurk for Summarization}, 
      author={Lining Zhang and Simon Mille and Yufang Hou and Daniel Deutsch and Elizabeth Clark and Yixin Liu and Saad Mahamood and Sebastian Gehrmann and Miruna Clinciu and Khyathi Raghavi Chandu and Jo{\~a}o Sedoc},
      year={2023},
      eprint={2212.10397},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```