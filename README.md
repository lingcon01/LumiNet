## Robust Protein-Ligand Interaction Modeling Integrating Physical Laws and Geometric Knowledge for Absolute Binding Free Energy Calculation


![](https://github.com/lingcon01/LumiScore/blob/master/SuScore/work_frame.png)

website: http://ai2physic.top

How to preprocess the protein and ligand data:
```
python ./SuScore/feats/extract_pocket_prody.py
python ./SuScore/feats/mol2radius.py
```


Use model to predict ABFE:
```
sh ./predict/run_suscore.sh
```

pretrain LumiScore:
```
python ./scripts/train_model.py
```

Semi-train LumiScore with fep+ dataset:
```
python ./scripts/semi_policy_train.py
```

train and test on PDE dataset:
```
python ./scripts/PDE_train.py
```

