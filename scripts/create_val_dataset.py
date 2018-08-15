import json

with open('../data/dataset_coco.json') as fid:
    coco = json.load(fid)
for image in coco['images']:
    filepath = image['filepath']
    if filepath == 'train2014':
        split = 'train'
    elif filepath == 'val2014':
        split = 'val'
    else:
        raise Exception()
    image['split'] = split
with open('../data/dataset_coco_val.json', 'w') as fid:
    json.dump(coco, fid)
print('finish')
