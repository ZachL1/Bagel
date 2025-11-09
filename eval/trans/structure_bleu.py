import easyocr
import os
from tqdm import tqdm
import argparse
import sacrebleu
import numpy as np
from shapely.geometry import Polygon
import json

language_map = {
            "aa":      "Afar",
            "ab":      "Abkhazian",
            "ae":      "Avestan",
            "af":      "Afrikaans",
            "ak":      "Akan",
            "am":      "Amharic",
            "an":      "Aragonese",
            "ar":      "Arabic",
            "as":      "Assamese",
            "av":      "Avaric",
            "ay":      "Aymara",
            "az":      "Azerbaijani",
            "ba":      "Bashkir",
            "be":      "Belarusian",
            "bg":      "Bulgarian",
            "bh":      "Bihari languages",
            "bi":      "Bislama",
            "bm":      "Bambara",
            "bn":      "Bengali",
            "bo":      "Tibetan",
            "br":      "Breton",
            "bs":      "Bosnian",
            "ca":      "Catalan",
            "ce":      "Chechen",
            "ch":      "Chamorro",
            "co":      "Corsican",
            "cr":      "Cree",
            "cs":      "Czech",
            "cu":      "Church Slavic",
            "cv":      "Chuvash",
            "cy":      "Welsh",
            "da":      "Danish",
            "de":      "German",
            "dv":      "Divehi",
            "dz":      "Dzongkha",
            "ee":      "Ewe",
            "el":      "Greek",
            "en":      "English",
            "eo":      "Esperanto",
            "es":      "Spanish",
            "et":      "Estonian",
            "eu":      "Basque",
            "fa":      "Persian",
            "ff":      "Fulah",
            "fi":      "Finnish",
            "fj":      "Fijian",
            "fo":      "Faroese",
            "fr":      "French",
            "fy":      "West Frisian",
            "ga":      "Irish",
            "gd":      "Gaelic",
            "gl":      "Galician",
            "gn":      "Guarani",
            "gu":      "Gujarati",
            "gv":      "Manx",
            "ha":      "Hausa",
            "he":      "Hebrew",
            "hi":      "Hindi",
            "ho":      "Hiri Motu",
            "hr":      "Croatian",
            "ht":      "Haitian",
            "hu":      "Hungarian",
            "hy":      "Armenian",
            "hz":      "Herero",
            "ia":      "Interlingua",
            "id":      "Indonesian",
            "ie":      "Interlingue",
            "ig":      "Igbo",
            "ii":      "Sichuan Yi",
            "ik":      "Inupiaq",
            "io":      "Ido",
            "is":      "Icelandic",
            "it":      "Italian",
            "iu":      "Inuktitut",
            "ja":      "Japanese",
            "jv":      "Javanese",
            "ka":      "Georgian",
            "kg":      "Kongo",
            "ki":      "Kikuyu",
            "kj":      "Kuanyama",
            "kk":      "Kazakh",
            "kl":      "Kalaallisut",
            "km":      "Central Khmer",
            "kn":      "Kannada",
            "ko":      "Korean",
            "kr":      "Kanuri",
            "ks":      "Kashmiri",
            "ku":      "Kurdish",
            "kv":      "Komi",
            "kw":      "Cornish",
            "ky":      "Kirghiz",
            "la":      "Latin",
            "lb":      "Luxembourgish",
            "lg":      "Ganda",
            "li":      "Limburgan",
            "ln":      "Lingala",
            "lo":      "Lao",
            "lt":      "Lithuanian",
            "lu":      "Luba-Katanga",
            "lv":      "Latvian",
            "mg":      "Malagasy",
            "mh":      "Marshallese",
            "mi":      "Maori",
            "mk":      "Macedonian",
            "ml":      "Malayalam",
            "mn":      "Mongolian",
            "mr":      "Marathi",
            "ms":      "Malay",
            "mt":      "Maltese",
            "my":      "Burmese",
            "na":      "Nauru",
            "nb":      "Norwegian Bokmål",
            "nd":      "North Ndebele",
            "ne":      "Nepali",
            "ng":      "Ndonga",
            "nl":      "Dutch",
            "nn":      "Norwegian Nynorsk",
            "no":      "Norwegian",
            "nr":      "South Ndebele",
            "nv":      "Navajo",
            "ny":      "Chichewa",
            "oc":      "Occitan",
            "oj":      "Ojibwa",
            "om":      "Oromo",
            "or":      "Oriya",
            "os":      "Ossetian",
            "pa":      "Punjabi",
            "pi":      "Pali",
            "pl":      "Polish",
            "ps":      "Pashto",
            "pt":      "Portuguese",
            "qu":      "Quechua",
            "rm":      "Romansh",
            "rn":      "Rundi",
            "ro":      "Romanian",
            "ru":      "Russian",
            "rw":      "Kinyarwanda",
            "sa":      "Sanskrit",
            "sc":      "Sardinian",
            "sd":      "Sindhi",
            "se":      "Northern Sami",
            "sg":      "Sango",
            "si":      "Sinhala",
            "sk":      "Slovak",
            "sl":      "Slovenian",
            "sm":      "Samoan",
            "sn":      "Shona",
            "so":      "Somali",
            "sq":      "Albanian",
            "sr":      "Serbian",
            "ss":      "Swati",
            "st":      "Southern Sotho",
            "su":      "Sundanese",
            "sv":      "Swedish",
            "sw":      "Swahili",
            "ta":      "Tamil",
            "te":      "Telugu",
            "tg":      "Tajik",
            "th":      "Thai",
            "ti":      "Tigrinya",
            "tk":      "Turkmen",
            "tl":      "Tagalog",
            "tn":      "Tswana",
            "to":      "Tonga",
            "tr":      "Turkish",
            "ts":      "Tsonga",
            "tt":      "Tatar",
            "tw":      "Twi",
            "ty":      "Tahitian",
            "ug":      "Uighur",
            "uk":      "Ukrainian",
            "ur":      "Urdu",
            "uz":      "Uzbek",
            "ve":      "Venda",
            "vi":      "Vietnamese",
            "vo":      "Volapük",
            "wa":      "Walloon",
            "wo":      "Wolof",
            "xh":      "Xhosa",
            "yi":      "Yiddish",
            "yo":      "Yoruba",
            "za":      "Zhuang",
            "zh":      "Chinese",
            "zu":      "Zulu",
            "hi-Latn": "Hindi Latin",
            "zh-Hant": "Traditional Chinese",
            "ceb":     "Cebuano",
        }
# 基于语言名称匹配创建映射
map_to_easyocr = {
    "af": "af",      # Afrikaans
    "ar": "ar",      # Arabic
    "as": "as",      # Assamese
    "az": "az",      # Azerbaijani
    "be": "be",      # Belarusian
    "bg": "bg",      # Bulgarian
    "bh": "bh",      # Bihari
    "bn": "bn",      # Bengali
    "bs": "bs",      # Bosnian
    "cs": "cs",      # Czech
    "cy": "cy",      # Welsh
    "da": "da",      # Danish
    "de": "de",      # German
    "en": "en",      # English
    "es": "es",      # Spanish
    "et": "et",      # Estonian
    "fa": "fa",      # Persian (Farsi)
    "fr": "fr",      # French
    "ga": "ga",      # Irish
    "hi": "hi",      # Hindi
    "hr": "hr",      # Croatian
    "hu": "hu",      # Hungarian
    "id": "id",      # Indonesian
    "is": "is",      # Icelandic
    "it": "it",      # Italian
    "ja": "ja",      # Japanese
    "kn": "kn",      # Kannada
    "ko": "ko",      # Korean
    "ku": "ku",      # Kurdish
    "la": "la",      # Latin
    "lt": "lt",      # Lithuanian
    "lv": "lv",      # Latvian
    "mi": "mi",      # Maori
    "mn": "mn",      # Mongolian
    "mr": "mr",      # Marathi
    "ms": "ms",      # Malay
    "mt": "mt",      # Maltese
    "ne": "ne",      # Nepali
    "nl": "nl",      # Dutch
    "no": "no",      # Norwegian
    "oc": "oc",      # Occitan
    "pi": "pi",      # Pali
    "pl": "pl",      # Polish
    "pt": "pt",      # Portuguese
    "ro": "ro",      # Romanian
    "ru": "ru",      # Russian
    "sk": "sk",      # Slovak
    "sl": "sl",      # Slovenian
    "sq": "sq",      # Albanian
    "sr": "rs_cyrillic",  # Serbian -> Serbian (cyrillic)
    "sv": "sv",      # Swedish
    "sw": "sw",      # Swahili
    "ta": "ta",      # Tamil
    "te": "te",      # Telugu
    "th": "th",      # Thai
    "tl": "tl",      # Tagalog
    "tr": "tr",      # Turkish
    "ug": "ug",      # Uyghur
    "uk": "uk",      # Ukrainian
    "ur": "ur",      # Urdu
    "uz": "uz",      # Uzbek
    "vi": "vi",      # Vietnamese
    "zh": "ch_sim", # Chinese -> Simplified Chinese
    "zh-Hant": "ch_tra",  # Traditional Chinese
}

def calculate_iou(box1, box2):
    """
    计算两个四边形的IoU值
    """
    # 将输入的四边形坐标转换为Polygon对象
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)

    # 计算交集区域的面积
    intersection_area = poly1.intersection(poly2).area

    # 计算两个四边形的面积
    area1 = poly1.area
    area2 = poly2.area

    # 计算并集区域的面积
    union_area = area1 + area2 - intersection_area

    # 计算IoU值
    iou = intersection_area / union_area

    # print("iou:", iou)
    return iou

def match_boxes(boxes1, boxes2, iou_threshold=0.5):
    """
    对两个图片中的文本框进行匹配
    """
    matches = []

    for i, box1 in enumerate(boxes1):
        max_iou = 0
        max_j = -1

        for j, box2 in enumerate(boxes2):
            iou = calculate_iou(box1, box2)

            if iou > max_iou:
                max_iou = iou
                max_j = j
        # print("max_iou:", max_iou)
        if max_iou > iou_threshold:
            matches.append((i, max_j))
        else:
            matches.append((i, -1))

    return matches

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_file", type=str, default="data/trans_data/annotations_with_paths.jsonl", help="Path to jsonl file containing image paths.")
    parser.add_argument("--ref_key", type=str, default="tgt_img_path", help="Key for reference image path in jsonl.")
    parser.add_argument("--eval_key", type=str, default="tgt_img_path", help="Key for evaluation image path in jsonl.")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold.")

    args = parser.parse_args()
    
    # Read jsonl file
    items = []
    with open(args.jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            items.append(json.loads(line.strip()))
    
    # Group items by target language to load easyocr reader efficiently
    lang_groups = {}
    for idx, item in enumerate(items):
        tgt_lang = item['tgt_lang']
        if tgt_lang not in map_to_easyocr:
            print(f"Warning: {tgt_lang} not in map_to_easyocr")
            continue
        tgt_lang = map_to_easyocr[tgt_lang]
        if tgt_lang not in lang_groups:
            lang_groups[tgt_lang] = []
        lang_groups[tgt_lang].append((idx, item))
    
    # Process each language group
    all_generate_result = [None] * len(items)
    all_ref_result = [None] * len(items)
    
    for lang, lang_items in lang_groups.items():
        print(f"Processing language: {lang}, {len(lang_items)} items")
        reader = easyocr.Reader([lang])
        
        for idx, item in tqdm(lang_items, desc=f"OCR for {lang}"):
            ref_file = item[args.ref_key]
            generate_file = item[args.eval_key]
            
            # Check if files exist
            if not os.path.exists(ref_file):
                print(f"Warning: Reference file not found: {ref_file}")
                all_generate_result[idx] = ""
                all_ref_result[idx] = ""
                continue
            if not os.path.exists(generate_file):
                print(f"Warning: Generated file not found: {generate_file}")
                all_generate_result[idx] = ""
                all_ref_result[idx] = ""
                continue
            
            # OCR
            generate_ocr_result = reader.readtext(generate_file, paragraph=True)
            ref_ocr_result = reader.readtext(ref_file, paragraph=True)
            generate_ocr_boxes = [item[0] for item in generate_ocr_result]
            ref_ocr_boxes = [item[0] for item in ref_ocr_result]

            matches = match_boxes(ref_ocr_boxes, generate_ocr_boxes, iou_threshold=args.iou_threshold)
            generate_ocr_result = [generate_ocr_result[item[1]][1] if item[1] != -1 else '' for item in matches]
            ref_ocr_result = [ref_ocr_result[item[0]][1] for item in matches]

            generate_text = ' '.join(generate_ocr_result)
            ref_text = ' '.join(ref_ocr_result)

            all_generate_result[idx] = generate_text
            all_ref_result[idx] = ref_text
    
    # calculate bleu
    bleu = sacrebleu.corpus_bleu(all_generate_result, [all_ref_result])
    print("iou threshold: {}".format(args.iou_threshold))
    print("structure sacrebleu: {}".format(bleu.score))