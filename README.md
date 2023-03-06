### Some notes (add more as you like):
- Date-time perferred ordering: day > asset > timeslot. 
- Don't forget to automate data processing pipeline for ease of real testing (e.g. a parsing function?)
- Maybe learning the MSE loss is too hard (predict future return exactly) How about we learn the probablity of going up and down?

### Issue: 
- Mini dataset - unnamed column

### Tips:
#### Merge a multi-indexed dataframe `df` with a single indexed one `s`
```python
assert df.index.names == ['day', 'asset']
assert s.index.name == 'day'
df.merge(s, left_on='day', right_index=True)
```
#### Maximize Correlation
minimize MSE with constraining the output variance to be the same as training output variance.


#### Record-Breaker Record:
1. Transformer (0.0412): seed 2023; eval_L:8; train_L:16; n_epoch10; feature:1024; lr 5e-4 cyclic; Eval: n_epoch:1
2. Transformer (0.0507): seed 2023; eval_L:8; train_L:16; n_epoch10; feature:2048; lr 5e-4 cyclic; Eval: n_epoch:2