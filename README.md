# BERT-CNN-Fine-Tuning-For-Hate-Speech-Detection-in-Online-Social-Media
A BERT-Based Transfer Learning Approach for Hate Speech Detection in Online Social Media

Article: https://arxiv.org/pdf/1910.12574.pdf

Dataset used: https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data

In this architecture shown in the figure, the outputs of all transformer encoders are used instead of using the output of the latest transformer encoder.
So that the output vectors of each transformer encoder are concatenated, and a matrix is produced. 
The convolutional operation is performed with a window of size (3, hidden size of BERT which is 768 in BERT-base model) and the maximum value is generated for each transformer encoder by applying max pooling on the convolution output. By concatenating these values, a vector is generated which is given as input to a fully connected network.
By applying softmax on the input, the classification operation is performed.

![Model](https://github.com/ZeroxTM/BERT-CNN-Fine-Tuning-For-Hate-Speech-Detection-in-Online-Social-Media/blob/main/Images/BertCNN.png =250x250)

 By Alaa Grable
   
   MIT LICENSE:
   
   Copyright (c) 2020 Alaa Grable

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
