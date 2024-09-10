# Description
When you have limited labeled data, pre-training is a common practice to make your neural network work better. For example, to efficiently train an image classifier, you probably should take a neural network pre-trained on ImageNet, and fine-tune it for your classification task (supervised pre-training). It has been recently shown that self-supervised learning (SSL) methods works as well as supervised pre-training on image classification tasks. However, there is no widely known strong evidence that SSL improves image segmentation.
The goal of the project is to find out which pre-training methods can improve image segmentation.
# Roadmap
- Literature review: find some papers (more citations - better) where people have already compared pre-trained models on image segmentation.
- Choose an image segmentation dataset, split it on train, val and test.
- Choose pre-training methods to benchmark. Choose not too large models, if computational resources are limited
	- `torchvision.models.resnet50`
	- https://github.com/mlfoundations/open_clip
	- https://github.com/google-research/simclr
- Train and eval them on the prepared dataset