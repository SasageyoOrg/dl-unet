<p align="center">
  <a href="" rel="noopener">
  <img width=200px height=200px src="https://camo.githubusercontent.com/306255f451a95302c2c819356243214c442f3d4f167bf0fe8ffc16e7fc0b957d/68747470733a2f2f692e696d6775722e636f6d2f6f5468554845692e706e67" alt="Project logo"></a>
</p>

<h1 align="center">DL-UNet: A new CNN for Median Nerve Segmentation</br><sub></sub></h1>

<div align="center">
  
![Jupyter](https://img.shields.io/badge/implementation-Jupyter-orange)
![UNIVPM](https://img.shields.io/badge/organization-UNIVPM-red)
![GitHub](https://img.shields.io/github/license/SasageyoOrg/ia-decision-tree?color=blue)
</div>

---

## 📝 Table of Contents
- [About](#about)
- [Project Topology](#project-topology)
- [CNN Architecture](#cnn-arch)
- [Results](#results)
- [Built Using](#built_using)
- [Authors](#authors)
- [Acknowledgments](#acknowledgement)

## 📋 About <a name = "about"></a>

Carpal tunnel syndrome is the compression of median nerve, is a condition that causes numbness, tingling, or weakness in the hand. It commonly occurs in individuals working in occupations that involve use of vibrating manual tools or tasks with highly repetitive and forceful manual exertion. CTS diagnosis is done with Ultrasound imaging by monitoring the movement of the nerve. Medical US image segmentation present several challenges such as low image quality, noise, diversity and data insufficiency. Over the past few years, several interesting Deep Learning based solutions were presented to overcome these challenges. 

In this work, we investigate a few of them, in particular Lightweight Unet and Double Unet, as well as how they behave when applied on our particular problem, that is the localization and segmentation of the median nerve. Last, but not least, we also propose our variant called <b>Double Lightweight Unet (DL-UNet)</b>, that integrates some of the features from the models mentioned above.

Project's paper: <a href="https://github.com/SasageyoOrg/cvdl-dl-unet/blob/main/dlunet_paper_short.pdf">DL-UNet: A Convolutional Neural Network for Median Nerve Segmentation</a>

## 🗂 Project Topology <a name="project-topology"></a>
```
|-- dataset
|   |-- images/...
|   |-- masks/...
|
|-- modules
|   |-- models/...
|   |   |-- unet.py
|   |   |-- lightweight_unet.py
|   |   |-- doubleunet.py
|   |   |-- dblunet.py
|   |-- lib.py
|   |-- metrics.py
|   |-- models.py
|   |-- plots.py
|   
|-- trained_models
|   |-- unet/...
|   |-- lunet/...
|   |-- dbunet/...
|   |-- dblunet/...
|
|-- dlunet_paper.pdf
|-- dlunet_paper_short.pdf
|-- archs_seg_mn.ipynb
|-- empty_archs_seg_mn.ipynb
|-- models_testing_seg_mn.ipynb
```

## 📊 CNN Architecture <a name="cnn-arch"></a>

Schema             |  Layer-level implementation
:-------------------------:|:-------------------------:
![](https://i.imgur.com/faV2M83.jpg)  |  ![](https://i.imgur.com/3evuXH1.png)

## 🔖 Results <a name = "results"></a>

<img src="https://i.imgur.com/NGCOXrt.png" />

## ⛏️ Built Using <a name = "built_using"></a>

- [Colab](http://colab.research.google.com)

## ✍️ Authors <a name = "authors"></a>

- Conti Edoardo [@edoardo-conti](https://github.com/edoardo-conti)
- Federici Lorenzo [@lorenzo-federici](https://github.com/lorenzo-federici)
- Melnic Andrian [@andrian-melnic](https://github.com/andrian-melnic)

## 🎉 Acknowledgements <a name = "acknowledgement"></a>

- [W001232] - Computer Vision e Deep Learning Class - Professor <a href="https://vrai.dii.univpm.it/emanuele.frontoni"><i>Emanuele Frontoni</i></a>


