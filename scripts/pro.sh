python prepro_labels_part.py --input_json ../data/dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk
# python scripts/prepro_reference_json.py --input_json ./data/ai_challenger/caption_train_annotations_20170902.json ./data/ai_challenger/caption_validation_annotations_20170910.json --output_json ./data/eval_reference.json
# python scripts/prepro_ngrams.py --input_json data/data_chinese.json --dict_json data/chinese_talk.json --output_pkl data/chinese-train --split train