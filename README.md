# ReNet-pytorch-keras-chapter3

The chapter3 of the segmentation network summary: 
### Combine other mature structures with segmentation networks.

External links: ReNet: A Recurrent Neural Network Based Alternative to Convolutional Networks [paper](https://arxiv.org/pdf/1505.00393.pdf).
Combining the Best of Convolutional Layers and Recurrent Layers: A Hybrid Network for Semantic Segmentation [paper](https://arxiv.org/pdf/1603.04871.pdf).
Inside-Outside Net: Detecting Objects in Context with Skip Pooling and Recurrent Neural Networks [paper](https://arxiv.org/pdf/1512.04143.pdf).

##### These papers' ideas are combining the basic network with RNN. The purpose is to use the memory characteristics of RNN to extract the context information, thus making up for the spatial invariance of CNN in the segmentation.

Here, I offer two versions of ReNet: pytorch and keras. The pytorch version was found and modified on github, while the keras version was written by individual to be integrated with another network of keras version. Both versions have been tested successfully.

### Environment: 
  
            Pytorch version >> 0.4.1; Keras version >> 2.2.4
             
## Notes
1. Through this code writing, I learned a lot through the code conversion between different frameworks, especially the writing of the custom layer in Keras (I think it is more troublesome.). Through the use of different frameworks, I prefer the Pytorch framework.
2. The code was integrated into the network ErfNet (We'll talk about it later) and successfully trained and tested.
