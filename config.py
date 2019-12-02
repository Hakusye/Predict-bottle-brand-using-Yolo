configurations = {
    "config":{
    "SEED" : 1234,
    "DATA_LOOT" : "/",
    "ROOT_IMAGES_PATH":"/home/deepstation/Shirae/MYolo/self_images/",
    "ROOT_IMAGES_PATH_VAL":"/home/deepstation/Shirae/MYolo/self_images_val/",
    "EXT":"png",
    "BATCH_SIZE" : 32,
    "IMAGE_SIZE" : 224,
    "DEVICE" : "cuda",
    "CLASSES" : 14,
    "CROSS_CLASSES":5,
    "MEAN" : (0.485,0.456,0.406),
    "STD" : (0.299,0.224,0.225),
    "MAX_EPOCH" : 6,
    "WEIGHT_DECAY" : 1e-3
    },

    "class_name" :{
       "aquarius" : 0,
       "ayataka_brown" : 1,
       "calpis": 2,
        "namacha" : 3,
       "tropicana" : 4,
       "ooi_ocha" : 5,
       "coca_cola" : 6,
       "ayataka" : 7,
       "dekavita" : 8,
       "pokari" : 9,
       "iemon" : 10,
       "genmai" : 11,
       "koicha" : 12,
       "natural_green" : 13
    },

    "rev_class_name" :[
        "aquarius",
        "ayataka_brown",
         "calpis",
        "namacha",
        "tropicana",
       "ooi_ocha",
       "coca_cola",
       "ayataka",
       "dekavita",
       "pokari",
       "iemon",
       "genmai",
       "koicha",
       "natural_green"
    ],
    "japanese_reading" :[
        "アクエリアス",
        "あやたかほうじちゃ",
        "カルピス",
        "なまちゃ",
        "トロピカーナ",
        "おーいおちゃ",
        "コカコーラ",
        "あやたか",
        "デカビタ",
        "ポカリスウェット",
        "いえもん",
        "あやたかげんまいちゃ",
        "こいちゃ",
        "サントリーのりょくちゃ"
    ]
}
