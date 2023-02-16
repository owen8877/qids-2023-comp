### Some notes (add more as you like):
- Date-time perferred ordering: day > asset > timeslot. 
- Don't forget to automate data processing pipeline for ease of real testing (e.g. a parsing function?)


### Issue: 
- Mini dataset - unnamed column

### Tips:
#### Merge a multi-indexed dataframe `df` with a single indexed one `s`
```python
assert df.index.names == ['day', 'asset']
assert s.index.name == 'day'
df.merge(s, left_on='day', right_index=True)
```