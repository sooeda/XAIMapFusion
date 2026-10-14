# О проекте

XAIMapFusion — исследовательский проект, объединяющий несколько методов объяснимого искусственного интеллекта (XAI) для построения консенсусных карт значимости при анализе МРТ.

Предлагаемый подход использует объединение областей по визуальному сходству с учетом согласованности, стабильности и достоверноси объяснений.

# Установка
git clone https://github.com/sooeda/XAIMapFusion.git
cd XAIMapFusion
pip install numpy pandas scikit-learn scipy scikit-image pillow matplotlib lime captum torch torchvision

 # Структура проекта
explainers.py - config.py — настройки проекта и параметры экспериментов.

model_and_data.py - загрузка данных и работа с моделью классификации МРТ.

pipeline.py - основной конвейер объединения карт значимости.

roi.py - выделение  ROI.

consensus.py - построение консенсусных карт значимости.

utils.py - вспомогательные функции для обработки данных и визуализации.

acc.py -  оценка качества классификации и объяснений.

run_base_average.py - запуск экспериментов с базовым усреднением карт.

run_found_hparams.py - запуск экспериментов с подобранными гиперпараметрами.

# About project

XAIMapFusion is a research project that combines multiple explainable artificial intelligence (XAI) methods to generate consensus saliency maps for brain MRI analysis.

The proposed approach merges regions based on visual similarity while taking into account the agreement, stability, and faithfulness of explanations.

# Installation
git clone https://github.com/sooeda/XAIMapFusion.git
cd XAIMapFusion
pip install numpy pandas scikit-learn scipy scikit-image pillow matplotlib lime captum torch torchvision

# Project structure
explainers.py - implementation of methods for explaining model predictions.

config.py - project settings and experiment parameters.

model_and_data.py - data loading and MRI classification model handling.

pipeline.py - main pipeline for merging saliency maps.

roi.py - region of interest (ROI) extraction.

consensus.py - generation of consensus saliency maps.

utils.py - utility functions for data processing and visualization.

acc.py - evaluation of classification and explanation quality.

run_base_average.py - runs experiments using baseline map averaging.

run_found_hparams.py - runs experiments with selected hyperparameters.
