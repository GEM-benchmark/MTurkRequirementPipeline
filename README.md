# A Needle in a Haystack: An Analysis of High-Agreement Workers on MTurk for Summarization
<p align="center">
  <a href="https://virtual2023.aclweb.org/paper_P3709.html"><img alt="ACL Publication" src="https://img.shields.io/badge/ACL-2023-green.svg" /></a>
  <a href="https://gem-benchmark.com/"><img alt="GEM Benchmark" src="https://img.shields.io/badge/GEM-benchmark-red.svg" /></a>
  <a href="https://github.com/GEM-benchmark/MTurkRequirementPipeline/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
  <a href="https://github.com/GEM-benchmark/MTurkRequirementPipeline/blob/main/figures/ACL_Paper_3709_Poster.pdf"><img alt="Poster" src="https://img.shields.io/badge/ACL-poster-purple.svg" /></a>
  <a href="https://github.com/GEM-benchmark/MTurkRequirementPipeline/blob/main/figures/ACL_Paper_3709_Slides.pdf"><img alt="Slides" src="https://img.shields.io/badge/ACL-slides-purple.svg" /></a>
</p>

This repository contains analysis scripts for our ACL 2023 paper "A Needle in a Haystack: An Analysis of High-Agreement Workers on MTurk for Summarization" [[ACL](https://virtual2023.aclweb.org/paper_P3709.html)][[arXiv](https://arxiv.org/abs/2212.10397)][[PDF](https://arxiv.org/pdf/2212.10397.pdf)].

<p align="center">
  <image src='figures/pipeline.png' width="300px"/>
</p>

<p align="center">
<em>
  Fig1: 
</em>
</p>

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
