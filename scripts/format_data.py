from datasets import load_dataset

ds = load_dataset("loukritia/science-journal-for-kids-data")

kids_abstract = ds['train']['Kids Abstract']

og_abstract = ds['train']['Abstract (Original academic paper)']

with open('/scratch4/mdredze1/icachol1/layscisum/targets/skj.target', 'w') as fout:
    for i in range(len(kids_abstract)):
        kids_abstract[i] = kids_abstract[i].replace('\n', ' ').strip()
    fout.write('\n'.join(kids_abstract))

with open('/scratch4/mdredze1/icachol1/layscisum/inputs/skj.inputs', 'w') as fout:
    for i in range(len(og_abstract)):
        og_abstract[i] = og_abstract[i].replace('\n', ' ').strip()
    fout.write('\n'.join(og_abstract))