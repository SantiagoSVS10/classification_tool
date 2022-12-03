# classification_tool
Grapich tool for ML  binary classification problems


Create an Anaconda enviroment
```bash
conda create -n classification_tool python=3.7.11
conda activate classification_tool
```
Locate the project's root directory and use pip to install the requirements 
```bash
pip install -r requirements.txt
```
Put the binary classification datasets in the datasets folder
```bash
└── data
    └── datasets
        ├── example_dataset_1
        │         ├──classA
        │         │     ├── image1.png
        │         │     ├── image2.png
        │         │     └──...
        │         └──classB
        │               ├── image1.png
        │               ├── image2.png
        │               └──...
        │         
        ├── example_dataset_1
        │         ├──classA
        │         │     ├── image1.png
        │         │     ├── image2.png
        │         │     └──...
        │         └──classB
        │               ├── image1.png
        │               ├── image2.png
        │               └──...

Locate the project's root directory and run the following script (always run the scripts from root directory):
```bash
python main_gui.py
```
