
# TODO


- [train] try unfreezing backbone
- [train] try training from checkpoint with lower lr
- [model] get model structure/something to give Fajir
- [visualize] random samples from dataset, show predicted vs actual labels
?- [train] from scratch (unfreeze backbone)

<!-- - [dataset] investigate dataset balance -->
<!-- - [test] get test results of pretrained weights on Emodataset -->
<!-- - [dataset] investigate valence/arousal values -->
<!-- - [question] think why large difference between valence and arousal CCC -->
<!-- - [train] see if train script actually used pretrained weights (__file__) -->


# Training

- fine-tune | 16 ep | lr=0.0003 [acc_0.7335]
    - fine-tune | 3 ep | unfreeze backbone | 
    - fine-tune | _ ep | lr=0.00005 | 



# Difference in Valence and Arousal

Possible causes:
- poor annotation quality for arousal
- arousal may be subtler and less consistenly expressed [citation?] 
- 
