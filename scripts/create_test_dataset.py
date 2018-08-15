import json

with open('../data/dataset_coco.json') as fid:
    coco = json.load(fid)
for image in coco['images']:
    filepath = image['filepath']
    if filepath == 'train2014' or filepath == 'val2014':
        image['split'] = 'train'
    else:
        raise Exception()

with open('../data/image_info_test2014.json') as fid:
    test_data = json.load(fid)
images = coco['images']
for data in test_data['images']:
    filename = data['file_name']
    imgid = len(images)
    cocoid = data['id']
    print('filename:', filename)
    print('imgid', imgid)
    print('cocoid', cocoid)
    image = {
        'filepath': 'test2014',
        'sentids': [],
        'filename': filename,
        'imgid': imgid,
        'split': 'test',
        'sentences': [],
        'cocoid': cocoid
    }
    images.append(image)
with open('../data/dataset_coco_test.json', 'w') as fid:
    json.dump(coco, fid)
print('finish')
