import streamlit as st
import os
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(BASE_DIR, "Metrics1.png")
img2_path = os.path.join(BASE_DIR, "Heatmap1.png")

# Текст для первой модели
text1 = """
## Введение в нейронные сети    
##### Проект: классификация изображений с использованием ResNet50

Модель 1: 

Dataset - [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification?select=seg_train)

###### Всего объектов
- **Учебный сет:** 14 034  
- **Валидационный сет:** 3 000  

| № | Класс     | Train | Train % | Valid | Valid % |
|---|-----------|-------|---------|-------|---------|
| 1 | buildings | 2191  | 15.61%  | 437   | 14.57%  |
| 2 | forest    | 2271  | 16.19%  | 474   | 15.80%  |
| 3 | glacier   | 2404  | 17.13%  | 553   | 18.43%  |
| 4 | mountain  | 2512  | 17.90%  | 525   | 17.50%  |
| 5 | sea       | 2274  | 16.21%  | 510   | 17.00%  |
| 6 | street    | 2382  | 17.00%  | 501   | 16.70% |

Метод обучения: Fine-tuning предобученной нейросети.  
Модель ResNet50 предобучена на данных ImageNet.  

Все сверточные слои зафиксированы (заморожены).  
Разморожен только классификационный слой (fully connected layer), адаптированный под 6 классов.
"""

st.markdown(text1)

# Показываем картинки первой модели
img = Image.open(img_path)
img2 = Image.open(img2_path)
st.image(img, caption="График обучения", use_container_width=True)
st.image(img2, width=400)

# Текст для второй модели
text2 = """
---

Модель 2:

Dataset - [Skin Cancer Detection](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign?datasetId=174469&searchQuery=pyt&select=test)

###### Всего объектов
- **Train:** 2 637  
- **Valid:** 660  

| № | Класс     | Train | Train % | Valid | Valid % |
|---|-----------|-------|---------|-------|---------|
| 1 | benign    | 1440  | 54.57%  | 360   | 54.55% |
| 2 | malignant | 1197  | 45.43%  | 300   | 45.45% |

* Benign — доброкачественные образования  
* Malignant — злокачественные образования (рак)  

Метод обучения: Fine-tuning предобученной нейросети.  
Модель ResNet50 предобучена на данных ImageNet.  
Для адаптации под бинарную классификацию разморожены последний сверточный блок (`layer4`) и классификационный полносвязный слой (`fully connected layer`).

"""

st.markdown(text2)

# Показываем картинки второй модели
BASE_DIR2 = os.path.dirname(os.path.abspath(__file__))
img_path2 = os.path.join(BASE_DIR2, "Metrics2.png")
img_path3 = os.path.join(BASE_DIR2, "Heatmap2.png")

img2 = Image.open(img_path2)
img3 = Image.open(img_path3)
st.image(img2, use_container_width=True)
st.image(img3, width=400)