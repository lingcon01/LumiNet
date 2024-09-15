## LumiScore: a distance distribution-based precise prediction of absolute binding free energy and visualization for atomic pair energy

![](https://github.com/lingcon01/LumiScore/blob/master/SuScore/frame.png)

website: http://ai2physic.top

How to preprocess the protein and ligand data:
```
sh ./predict/run_suscore.sh

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

