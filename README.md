## Robust Protein-Ligand Interaction Modeling Integrating Physical Laws and Geometric Knowledge for Absolute Binding Free Energy Calculation![image](https://github.com/user-attachments/assets/cd50d1f9-4eb3-4ad0-b3b5-c8c662fadb7e)


![](https://github.com/lingcon01/LumiScore/blob/master/SuScore/frame.png)

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

finetune LumiScore with PDBbind2020:
```
python ./scripts/SuScore_train.py
```

Semi-train LumiScore with fep+ dataset:
```
python ./scripts/semi_policy_train.py
```

train and test on PDE dataset:
```
python ./scripts/PDE_train.py
```

