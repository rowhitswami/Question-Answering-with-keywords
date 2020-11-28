# Question-Answering-with-keywords
Building a scalable semantic search engine using Haystack, Elasticsearch and Transformers.

## Jupyter Notebook
Run the notebook on Kaggle with GPU:

[https://www.kaggle.com/rowhitswami/question-answering-with-keywords](https://www.kaggle.com/rowhitswami/question-answering-with-keywords)

## 1. Approach of the solution
* **Which ML/DL model architecture you used and why?**

I used the architecture of Albert (albert_xxlargev1_squad2_512) which is fine-tuned on SQuAD2. Stanford Question Answering Dataset (SQuAD) is simply a reading comprehension dataset which is nowadays used to train machine learning models. The dataset consists of questions raised over a certain paragraph by a community of crowd workers through Wikipedia articles. I chose Albert because it is large, powerful, SOTA model which provides better accuracy than any other open source model in Question Answer domain.

* **How would you ensure the scalability of your solution?**

The answer is **Haystack** and **Elasticsearch**.
Haystack is an open-source framework for building end-to-end question answering systems for large document collections. Recent advances in NLP have enabled the application of QA to real world settings and Haystack is designed to be the bridge between research and industry. It also ensure production-ready deployments that scale to millions of documents. While Elasticsearch is a distributed, open source search and analytics engine for all types of data, including textual, numerical, geospatial, structured, and unstructured.


Haystack is powered by a Retriever-Reader pipeline in order to optimise for both speed and accuracy.

![](https://haystack.deepset.ai/static/7cbd0592b7e0de24ec3f5b3628ca24ae/7a4b2/retriever_reader.png)

**Readers**, also known as Open-Domain QA systems in Machine Learning speak, are powerful models that do close analysis of documents and perform the core task of question answering. The Readers in Haystack are trained from the latest transformer based language models and can be significantly sped up using GPU acceleration. However, it is not currently feasible to use the Reader directly on large collection of documents.

The **Retriever** assists the Reader by acting as a lightweight filter that reduces the number of documents that the Reader has to process. It does this by:

* Scanning through all documents in the database
* Quickly identifying the relevant and dismissing the irrelevant
* Passing on only a small candidate set of documents to the Reader

* **Is there a need for any dataset? If yes then how much data is sufficient to train the model in order to get required results?**

Yes, we do need to store a corpus of articles in Elasticsearch server. Having said that, having more quality data is the only secret sauce of obtaining desired results in machine learning domain.
I used **7,241** research papers published in Neural Information Processing Systems (NIPS) conference from 1987 to 2017. It covers topics ranging from deep learning and computer vision to cognitive science and reinforcement learning.
Since, I am impressed with the answers extracted from articles, I am assuming having ~7000 documents for each domain with character limit of ~10,000 length is enough to make general prediction.

* **Is there a need to create manual datasets, if yes then what parameters and sample size did you consider to create a dataset?**

Not required.

* **Are your model and dataset generalized enough for different domains of the use cases, How?**

Yes, the Albert model which is fine-tuned on SQuAD2.0 comes with the combination of 100,000 answerable questions from SQuAD1.1 and the additional 50,000 unanswerable opposed questions. These unanswerable adversary questions are raised more similar to those of the answerable ones. So in order to perform excellently on SQuAD2.0, the trained machine learning systems must not only answer the answerable questions but also have to decide in no time whether the specific question can be answered or not from the given paragraph.If found unanswerable, it should quickly switch to the other question pointing that particular question is unanswerable from the paragraph.

This is the icebreaking significance of SQuAD 2.0. It trains the machine learning models not only to answer the questions from reading comprehension but also to abstain from unanswerable questions or adversities of the real world. Though SQuAD is composed entirely of Wikipedia articles, these models are flexible enough to deal with many different styles of text.

* **How would you train, test and deploy your model to production?**

We can wrap our model in a Flask/Django API and host it over EC2 instance of AWS. (Have experience in deploying the NLP solution to production)


* **How would you perform hyperparameter tuning on your model to improve accuracy?**

Though, not required in our case, but to perform hyperparameter tuning, we need define the range of possible values for all hyperparameters, an evaluative criteria to judge the model, and a cross-validation method. We can then use Grid Search, Random Search or Baysian Optimization technique to perform hyperparameter tuning depending on the problem. 


* **Anything else you want to let us know about your approach.**

1. Character limit - One of limitation of using Transformers is the fixed size of a document to process. The average characters in a document were ~25,000 but to use the Haystack architecture more efficiently I had to trim the length of document to average ~10,000 characters. The documents are trimmed in such a fashion that it respect the boundary of the sentence in a document, so that it doesn't loose the context.
2. Alternative architecture - I had a tough time in running the model in my local machine. Using **ahotrod/albert_xxlargev1_squad2_512** model is only possible if we are doing all the computations on a dedicated GPU. While running on CPU, please use **deepset/roberta-base-squad2**, which gives a good balance between speed and accuracy.

## 2. Approach to generate questions

This project has been in my to-do list for quite a while. I proposed to do question-answering generation system in my previous organization, but since my internship completion period was approaching I couldn't do it.
While researching I came up with this general approach for building a MCQs type question-answer generation system:
1. Let's say, we have a target sentence.

```
The color of apple is red.
```

The first thing could be to identify top-keywords from the passage and use them as answers to the questions. Let's say identified word **red** as our answer.
```
The color of apple is *red*.
```

2. Then we can replace the potential keyword (**red**) with a blank-space or underline.

```
The color of apple is ____.
```

3. The last step is to create distractors, the incorrect answers. We can find the most closest/similar word to our potential keyword using word embeddings and cosine similarity.
Expected semantically similar keywords to **red** in a vector space : Green, Orange, Blue.

4. Finally, we can compile everything.

```
The color of apple is ____.

1. Green 2. Red 3. Orange 4. Blue
```

