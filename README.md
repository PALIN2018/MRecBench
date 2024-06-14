# MRecBench: Benchmarking Large Vision Language Models on Multimodal Recommendation

This repository includes the dataset and benchmark of the paper:

**MRecBench: Benchmarking Large Vision Language Models on Multimodal Recommendation (Submitted to NeurIPS 2024 Track on Datasets and Benchmarks).**

**Authors**: Peilin Zhou, Chao Liu, Jing Ren, Xinfeng Zhou, Yueqi Xie, Meng Cao, You-Liang Huang, Dading Chong, Guojun Yin, Wei Lin, Junling Liu, Jae Boum KIM, Shoujin Wang, Raymond Chi-Wing Wong, Sunghun Kim

## Abstract
Large Vision Language Models (LVLMs) have demonstrated considerable potential in various tasks that require the integration of visual and textual data. However, their employment in multimodal recommender systems (MRSs) has not been thoroughly investigated though they can play a significant role to greatly improve the performance of multimodal recommendations. To address this gap, we present MRecBench, the first comprehensive benchmark to systematically evaluate different LVLM integration strategies in recommendation scenarios. We benchmark three state-of-the-art LVLMs, i.e., GPT-4 Vision, GPT-4o, and Claude3-Opus, on the next item prediction task using the constructed Amazon Review Plus dataset, which includes additional item descriptions generated by LVLMs. Our evaluation focuses on five integration strategies: using LVLMs as recommender, item enhancer, reranker, and a combination of these roles. The findings from MRecBench provide critical insights into the performance disparities among these strategies, highlight the most effective methods for leveraging LVLMs in multimodal recommendations, and offer guidance for future research and practice in MRSs. All benchmark results, codes, and datasets are available in [link](https://github.com/PALIN2018/MRecBench). We hope this benchmark can serve as a critical resource for advancing the research in the area of MRSs.

## Dataset
We constructed the Amazon Review Plus Dataset by augmenting the original Amazon Review dataset (McAuley et al.) using LVLMs. The dataset focuses on four categories: beauty, sports, toys, and clothing. For each item image in each category, we employed LVLMs to describe the content in the image. The generated image captions are considered extra item side information and combined with the original dataset.

## Get Started
To get started with MRecBench, follow these steps:

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- PyTorch
- Transformers
- Other dependencies listed in `requirements.txt`

### Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/PALIN2018/MRecBench.git
cd MRecBench
pip install -r requirements.txt
```
### Obtaining S1 Results:
1. Install dependencies according to `requirements.txt`.

2. Modify the file paths in the `.ipynb` file as needed.

3. Run the `.ipynb` file directly.

### Obtaining S2 Results:
1. Navigate to the `sasrec` folder.
2. Follow the instructions in the `README.md` file to run the code.

### Obtaining S3, S4, S5 Results:
1. Install dependencies according to `requirements.txt`.

2. Modify the paths in the `run.sh` file as instructed.

3. Run the `run.sh` file.

## Contact

If you have any questions about our dataset or benchmark results, please feel free to contact us!
(Peilin Zhou, zhoupalin@gmail.com)