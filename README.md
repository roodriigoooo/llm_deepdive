## deep dive into large language models

An exploration of Natural Language Processing, from introductory exploration to the development of an LLM. My goal is to explore traditional GPT2-3 transformer architectures, more recent Llama architectures, and eventually dive into developments such as Latent Attention, PPO, GRPO, and RLHF. This is a personal effort to study and understand the architectures, techniques, concepts, and trends surrounding the field of LLMs, and to feel comfortable implementing them in a variety of contexts. 

## references and datasets used

Raschka, S. (2024). *Build a Large Language Model (From Scratch)*. Manning. ISBN: 978-1633437166.

> *Build a Large Language Model (From Scratch)*, authored by Sebastian Raschka, forms the basis of this exploration, and was the work that originally sparked my interest, perhaps because it offered explanations to concepts (which I once found too complex) with such clarity that they eventually became genuinely understandable.

Zhang, A., Lipton C.Z., Li, M., Smola, J.A. (2023). *Dive into Deep Learning*. Cambridge

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, N.A., Kaiser, L., Polosukhin, I. (2017, revised 2023). Attention is All You Need. arXiv:1706.03762. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

Lei, B.J., Kiros, R.J., Hinton, E.G., (2016). Layer Normalization. arXiv:1607.06450. [https://arxiv.org/abs/1607.06450](https://arxiv.org/abs/1607.06450)

Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan, Y., Wang, L., Liu, T. (2020). On Layer Normalization in the Transformer Architecture. arXiv:2002.04745. [https://arxiv.org/pdf/2002.04745](https://arxiv.org/pdf/2002.04745)

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. *OpenAI* [https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

Sennrich, R., Haddow, B., Birch, A., (2015). Neural Machine Translation of Rare Words with Subword Units. arXiv:1508.07909. [https://arxiv.org/pdf/1508.07909](https://arxiv.org/pdf/1508.07909)

Karpathy, A. (2024, Feb 20). *Lets build the GPT Tokenizer*. YouTube. [Lets build the GPT tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)

> This video by Andrej Karpathy (and all the others in his *Neural Networks: Zero to Hero* playlist) implements, from scratch the Byte-Pair encoding. The code contained in my notebook implements some modifications and is in a way more exploratory, but all the logic and main implementation ideas are all his. I can't recommend enough the resources he puts out for free, especially coming from one of the top AI experts. 

#### datasets

Austen, J. (2010). Pride and Prejudice. [Project Gutenberg](https://www.gutenberg.org/)

Chopin, K. (2006). The Awakening, and Selected Short Stories by Kate Chopin. [Project Gutenberg](https://www.gutenberg.org/)

Allison Parrish's Gutenberg Poetry Corpus

> [This dataset](https://github.com/aparrish/gutenberg-poetry-corpus) was the one I used for the Byte-Pair tokenization training. A great dataset containing more than 3 million lines of poetry, also extracted from books from the Project Gutenberg.