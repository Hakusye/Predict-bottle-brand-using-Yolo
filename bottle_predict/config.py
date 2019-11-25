configurations = {
    "config":{
    "SEED" : 1234,
    "DATA_LOOT" : "/",
    "ROOT_IMAGES_PATH":"/home/deepstation/Shirae/MYolo/self_images/",
    "EXT":"png",
    "BATCH_SIZE" : 32,
    "IMAGE_SIZE" : 224,
    "DEVICE" : "cuda",
    "CLASSES" : 14,
    "CROSS_CLASSES":5,
    "MEAN" : (0.485,0.456,0.406),
    "STD" : (0.299,0.224,0.225),
    "MAX_EPOCH" : 5,
    "WEIGHT_DECAY" : 1e-3
    },

    "class_name" :{
       "aquarius" : 0,
       "ayataka_brown" : 1,
       "calpis": 2,
        "namacha" : 3,
       "natural_green" : 4,
       "tropicana" : 5,
       "ooi_ocha" : 6,
       "coca_cola" : 7,
       "ayataka" : 8,
       "dekavita" : 9,
       "pokari" : 10,
       "iemon" : 11,
       "genmai" : 12,
       "koicha" : 13
    },

    "rev_class_name" :[
        "aquarius",
        "ayataka_browm",
        "calpis",
        "namacha",
        "natural_green",
        "tropicana",
       "ooi_ocha",
       "coca_cola",
       "ayataka",
       "dekavita",
       "pokari",
       "iemon",
       "genmai",
       "koicha"
    ]
}
