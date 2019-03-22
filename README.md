# Histopathologic-Cancer-Detection
https://www.kaggle.com/c/histopathologic-cancer-detection


## Linux查看Nvidia显卡信息及使用情况

Nvidia自带一个命令行工具可以查看显存的使用情况：

```
nvidia-smi
```

## 模型

`model.json`
```
"ResNet9", "CNN"
```

`training.json`
```
"lr_method": "Adam","Adamax",
"criterion_method": "CrossEntropyLoss", 'MSELoss', 'BCEWithLogitsLoss'
```
`ResNet9` 一定要用 `BCEWithLogitsLoss`
`CNN` 一定要用 `CrossEntropyLoss`
