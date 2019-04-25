# Bachelor Draft

In order to provide more transparency on convolutional neural networks (CNN) for education and
research purposes this work has three main objectives:

- Create CNN models based on two different architectures, which classify magnetic
resonance images of the brain into normal and abnormal.<br />
- Investigate these models by applying state of the art visualization techniques. For this
purpose, a simple accessible Application Interface (API) for Keras sequential models
will be developed.<br />
- Describe the quality of the models based on the visualizations and compare their
overall classification to the human classification procedure.<br />


![Learning](https://github.com/JakobDexl/Bachelor/blob/master/Test_visulizations/stack2.gif) <br />
*the animation above shows the progression of the <br />
activation maps of the first layer during training*

## Purpose

In recent years the number of prescribed computed tomography (CT) and magnetic resonance (MR)
admissions has continuously increased, a trend, which is observed over all OECD countries. Since 2006,
the average number of CT and MR examinations per 1000 people increased by 34% and 22%,
respectively, mainly affected are the United States, Japan and Germany (Appendix A, Table 8 and Table
9) [1]. E. g. the Mayo Clinic in Minnesota, USA recorded an increase of 68% and 85% of CT and MRI
examinations over one decade. Simultaneously, the amount of data being produced per examination
also increased. In 1999 an average CT exam consisted of 82 images, whereas in 2010 it contained up
to 679 images. Likewise, the average number of slices obtained by one MRI scan has risen from 164 to
570 images. Accordingly, the radiologists are facing an increasing workload with a time frame of only
3-4 seconds per scan [2].
One third of all investigations are focussed on the brain (Appendix A, Fig.3) [3]. However, internal
studies have shown that most of the findings in brain MR images are normal (Appendix A, Fig.2)
[4]. As a result, physicians are losing a lot of time and energy for non-pathological indications.
Computer Aided Diagnosis systems (CAD) that apply machine learning/intelligent algorithms to
automatically classify radiologic data [5,6] could therefore provide significant workflow improvement.
The superordinate field of research is called Radiomics. It aims at gaining insights from huge image
databases [7,8]. In literature three strategies are discussed to accomplish the described task (Figure
1): the more classical Feature Engineering (blue), the newer Deep Learning (DL) approach (orange), as
well as their combination (green).
||
Figure 1 shows the concept of a computer aided diagnosis system, and possible approaches to fulfil a classification task
like classifying brains into normal and abnormal.
Hochschule Landshut – Bachelorarbeit – Jakob Dexl
Seite 7The Feature Engineering approach is overall more comparable to the human classification procedure.
One or more important target biomarkers are selected, pre-processed and quantified. Thus, arbitrary
markers can be obtained. The extracted or selected biomarkers serve as features that are used for
classification using machine learning models. State of the art results are reaching up to 100% accuracy
and can be achieved through methods like Discrete Wavelet Transformations (DWT) for feature
extraction, Principle Component Analysis (PCA) for feature selection and Support Vector Machines for
classification (SVM) [9–11,5]. Advantages are the control of the marker extraction and hence the
justification of the classification results. The feature extraction requires a deep knowledge of
mathematics and medicine. Therefore, the procedures are difficult to generalize, which is a major
disadvantage in case of inhomogeneous medical image data.
The second subfield deals with Deep Learning models which have become popular in the last few years.
These won numerous non-medical classification competitions and outperform classic feature
engineering [12–14]. Models automatically extract features based on a data set. Possible medical
applications are ranging from classification, segmentation or pre-processing tasks and are currently an
intense field of research [15]. Recent papers demonstrate this potential with good results for the
described image classification task [16–18]. Limiting factors include the lack of data and the high data
heterogeneity. For example the companies Nvidia, GE and Nuance currently announced a partnership
to collect CT image data for their DL platforms [19]
Another big challenge is the loss of transparency due to fact that these systems cannot easily be
dismantled into their components, which often causes model mistrust. Indeed, DL models are not
intuitively interpretable for humans as long as they are not providing any justification on themselves
[20]. To fight this lack of transparency for humans it is highly important to provide explanations on
model decision making systems. This could convey model trust for developers, regulators, users and
provides interpretability for them [21]. In addition, visualizations could be used for research to gain
new insights from unknown data, for example to provide hints for better medical biomarkers [22] or
to disprove misleading ones [23]. Also visualization techniques are used for educational purposes in
many blogs and books [24]. Finally, visualizations are necessary to meet the high standards of medical
technologies and to be in compliant with EU law. This year (2018) a new EU data regulation entered
into force, which garants consumers the ‘right to explanation’ on algorithmic decision making systems
[25].

### Reference

1  OECD.Stat: Health Care Statistics: OECD Publishing, 2018<br />
2  McDonald RJ, Schwartz KM, Eckel LJ et al. The effects of changes in utilization and technological
advancements of cross-sectional imaging on radiologist workload. Academic radiology 2015; 22:
1191 – 1198<br />
3  BARMER. (n.d.). Anzahl der CT- und MRT-Untersuchungen in Deutschland nach Bereich 2009 (in
Millionen), https://bibaccess.fh-landshut.de:2127/statistik/daten/studie/172722/umfrage/ct-
und-mrt---untersuchungen-nach-bereich-2009/<br />
4  Regina Felix. Nicht-Normalbefunds-Warnung in der Radiologie – Konzeptionierung einer
technischen Lösung: Hochschule Landshut, 2017<br />
5  Jha D, Kim J-I, Choi M-R et al. Pathological Brain Detection Using Weiner Filtering, 2D-Discrete
Wavelet Transform, Probabilistic PCA, and Random Subspace Ensemble Classifier. Computational
intelligence and neuroscience 2017; 2017: 4205141<br />
6  Wang H, Ahmed SN, Mandal M. Computer-aided diagnosis of cavernous malformations in brain
MR images. Computerized medical imaging and graphics : the official journal of the
Computerized Medical Imaging Society 2018; 66: 115 – 123<br />
7  Choi JY. Radiomics and Deep Learning in Clinical Imaging: What Should We Do? Nuclear medicine
and molecular imaging 2018; 52: 89 – 90<br />
8  Kai C, Uchiyama Y, Shiraishi J et al. Computer-aided diagnosis with radiogenomics: Analysis of
the relationship between genotype and morphological changes of the brain magnetic resonance
images. Radiological physics and technology 2018<br />
9  Saritha M, Paul Joseph K, Mathew AT. Classification of MRI brain images using combined wavelet
entropy based spider web plots and probabilistic neural network. Pattern Recognition Letters
2013; 34: 2151 – 2156<br />
10 Siddiqui MF, Reza AW, Kanesan J. An Automated and Intelligent Medical Decision Support
System for Brain MRI Scans Classification. PLoS ONE 2015; 10<br />
11 Kumar S, Dabas C, Godara S. Classification of Brain MRI Tumor Images: A Hybrid Approach.
Procedia Computer Science 2017; 122: 510 – 517<br />
12 Krizhevsky A, Sutskever I, Hinton GE. ImageNet Classification with Deep Convolutional Neural
Networks, 2012: 1097 – 1105<br />
13 He K, Zhang X, Ren S et al. Deep Residual Learning for Image Recognition<br />
14 Simonyan K, Zisserman A. Very Deep Convolutional Networks for Large-Scale Image Recognition<br />
15 Shen D, Greenspan H, Zhou SK (eds). Deep learning for medical image analysis. London, United
Kingdom: Academic Press, 2017<br />
16 Mohsen H, El-Dahshan E-SA, El-Horbaty E-SM et al. Classification using deep learning neural
networks for brain tumors. Future Computing and Informatics Journal 2017<br />
17 Rezaei M, Yang H, Meinel C. Deep Learning for Medical Image Analysis<br />
18 Payan A, Montana G. Predicting Alzheimer's disease: A neuroimaging study with 3D
convolutional neural networks (2015/02/09), 2015, http://arxiv.org/pdf/1502.02506<br />
19 NVIDIA, GE Healthcare, Nuance to Bring Power of AI to Medical Imaging | NVIDIA Blog, 2018,
https://blogs.nvidia.com/blog/2017/11/26/ai-medical-imaging/<br />
20 Lipton ZC. The Mythos of Model Interpretability (2017/03/06), 2017,
http://arxiv.org/pdf/1606.03490<br />
21 Selvaraju RR, Cogswell M, Das A et al. Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization (2017/03/21), 2017, http://arxiv.org/pdf/1610.02391<br />
22 Kamnitsas K, Ledig C, Newcombe VFJ et al. Efficient Multi-Scale 3D CNN with Fully Connected
CRF for Accurate Brain Lesion Segmentation (2017/01/08), 2017,
http://arxiv.org/pdf/1603.05959<br />
23 Stefan H, Frank M, Kai G et al. On the interpretation of weight vectors of linear models in
multivariate neuroimaging. NeuroImage 2014; 87: 96 – 110<br />
24 Chollet F. Deep learning with Python. Shelter Island, NY: Manning Publications Co, 2018
25 Goodman B, Flaxman S. European Union regulations on algorithmic decision-making and a "right
to explanation" (2016/08/31), 2016, http://arxiv.org/pdf/1606.08813<br />


### Blogs

- [Understanding and Visualizing Convolutional Neural Networks ](http://cs231n.github.io/understanding-cnn/) (1,2,3,4,5) <br />
- [Feature Visualization - How neural networks build up their understanding of images](https://distill.pub/2017/feature-visualization/#enemy-of-feature-vis) (1,2) <br />
- [Deepvis](http://yosinski.com/deepvis) (1,2,4,5) <br />
- [The Building Blocks of Interpretability](https://distill.pub/2018/building-blocks/) (2, 4) <br />
- [Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/) (5) <br />

### Misc

- [Data Preprocessing](http://cs231n.github.io/neural-networks-2/) <br />

### Illustration directory
(1) Thanks to Dr. Vemuri for her permission to took the Image out of this work
 [Vemuri, Prashanthi; Jack, Clifford R. (2010): Role of structural MRI in Alzheimer's disease. In: Alzheimer's Research & Therapy 2 (4), S. 23. DOI: 10.1186/alzrt47.](https://alzres.biomedcentral.com/articles/10.1186/alzrt47)


